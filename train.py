"""
Retrain the YOLO model for your own dataset.
"""

import numpy as np
# import os
# os.environ['KERAS_BACKEND']='tensorflow'
# from tensorflow.python.keras._impl.keras import backend as K
# from keras import backend as K
# from tensorflow.keras.layers import Input, Lambda
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss # 1,yolo_loss2 # yolo_loss1为原始yolo_v3 loss;yolo_loss2为高斯yolo_v3 loss
from yolo3.utils import get_random_data


def _main():
    annotation_path = 'data/0219/train.txt'        # train.txt；data/0219/train.txt；data/0319/train.txt
    log_dir = 'logs/006/'
    classes_path = 'model_data/gw0219_classes.txt' # model_data/voc_classes.txt；model_data/gw0219_classes.txt；model_data/flower0319_classes.txt
    anchors_path = 'model_data/yolo_anchors_gw0219.txt'   # model_data/yolo_anchors.txt；model_data/yolo_anchors_gw0219.txt；model_data/yolo_anchors_flower0319.txt
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    # class_weighting系数
    class_weighting = [2.26, 2.33, 1.00, 1.84, 1.42] ###
    # from sklearn.utils import class_weight
    # class_weighting = class_weight.compute_class_weight('balance',np.unique(train_generator.classes),train_generator.classes)
    # class_weighting = 'auto'
    # class_weighting = 'balanced'


    input_shape = (416,416) # multiple of 32, hw

    is_tiny_version = len(anchors)==6 # default setting
    if is_tiny_version:
        model = create_tiny_model(input_shape, anchors, num_classes,
            freeze_body=2, weights_path='model_data/tiny_yolo_weights.h5')
    else:
        model = create_model(input_shape, anchors, num_classes,
            freeze_body=2, weights_path='model_data/yolov3_weights.h5') # make sure you know what you freeze h5变更

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=False, save_best_only=True, period=3) # 原save_weights_only=False = True，
    # True：保存模型参数，但不保存模型结构，False：它既保持了模型的图结构，又保存了模型的参数
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)  #
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if True:
        model.compile(optimizer=Adam(lr=1e-3), loss={ # lr=1e-3
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})

        batch_size = 16  # 原32
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
                validation_steps=max(1, num_val//batch_size),
                epochs=25, # 50
                initial_epoch=0,
                callbacks=[logging, checkpoint],
                class_weight=class_weighting)
        # model.save_weights(log_dir + 'trained_weights_stage_1.h5') # 原
        model.save(log_dir + 'trained_weights_stage_1.h5')
    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
        print('Unfreeze all of the layers.')

        batch_size = 2 # note that more GPU memory is required after unfreezing the body 原32 笔记本硬件条件，batch_size只能选择2
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val//batch_size),
            epochs=45, # 100
            initial_epoch=25,# 50
            callbacks=[logging, checkpoint, reduce_lr, early_stopping],# 终止训练只在最后一步进行
            class_weight=class_weighting)
        # model.save_weights(log_dir + 'trained_weights_final.h5') # 原
        model.save(log_dir + 'trained_weights_final.h5')

    # Further training if needed. 如果需要进一步训练


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/yolo.h5'):
    '''create the training model'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)
    # y_true变化形式如下：
#   fm_13_input = Input(shape=(None, None, num_anchors//3, num_classes + 5), name='fm_13_input')
#   fm_26_input = Input(shape=(None, None, num_anchors//3, num_classes + 5), name='fm_26_input')
#   fm_52_input = Input(shape=(None, None, num_anchors//3, num_classes + 5), name='fm_52_input')
    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], num_anchors//3, num_classes+5)) for l in range(3)]
    # h//{0:32, 1:16, 2:8}[l]代表字典为0时，h//32；字典为1时，h//16；字典为2时，h//8
    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    # if load_pretrained: # 装载预训练
    #     model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
    #     # model_body.load_weights(weights_path, by_name=True)
    #     print('Load weights {}.'.format(weights_path))
    #     if freeze_body in [1, 2]:
    #         # Freeze darknet53 body or freeze all but 3 output layers.
    #         num = (185, len(model_body.layers)-3)[freeze_body-1]
    #         for i in range(num): model_body.layers[i].trainable = False
    #         print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss) # 函数式模型接口：model = Model(inputs=[a1, a2], outputs=[b1, b3, b3])

    if load_pretrained: # 装载预训练
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        # model_body.load_weights(weights_path, by_name=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    return model

def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/tiny_yolo_weights.h5'):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], \
        num_anchors//2, num_classes+5)) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors//2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze the darknet body or freeze all but 2 output layers.
            num = (20, len(model_body.layers)-2)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator''' # fit_generator数据生成器
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines) # 在一个batch中进行序号洗牌
            image, box = get_random_data(annotation_lines[i], input_shape, random=True) # 随机改变图像效果，输出改变后的图像与相应box坐标
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data) # 矩阵化image数据
        box_data = np.array(box_data)     # 矩阵化box数据
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes) # 为每个真实框体找到最合适的anchor（9个中选1）
        yield [image_data, *y_true], np.zeros(batch_size) # yield作用：每个batch都按照上次结束位置再往下进行；

def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes): # 数据发生器包装器
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)

if __name__ == '__main__':
    _main()
