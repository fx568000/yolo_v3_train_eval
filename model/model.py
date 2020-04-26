"""YOLO_v3 Model Defined in Keras."""

from functools import wraps

import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2

from yolo3.utils import compose
from yolo3.util_graphs import box_iou_graph, y_pred_graph

def nll_loss(x, mu, sigma, sigma_const=0.3):
    pi = tf.constant(np.pi)
    Z = (2 * pi * (sigma + sigma_const) ** 2) ** 0.5
    probability_density = tf.exp(-0.5 * (x - mu) ** 2 / ((sigma + sigma_const) ** 2)) / Z
    nll = -tf.log(probability_density + 1e-7)
    return nll

@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)

def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))

def resblock_body(x, num_filters, num_blocks):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3,3), strides=(2,2))(x)
    for i in range(num_blocks):
        y = compose(
                DarknetConv2D_BN_Leaky(num_filters//2, (1,1)),
                DarknetConv2D_BN_Leaky(num_filters, (3,3)))(x)
        x = Add()([x,y])
    return x

def darknet_body(x):
    '''Darknent body having 52 Convolution2D layers'''
    x = DarknetConv2D_BN_Leaky(32, (3,3))(x)
    x = resblock_body(x, 64, 1)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    x = resblock_body(x, 512, 8)
    x = resblock_body(x, 1024, 4)
    return x

def make_last_layers(x, num_filters, out_filters):
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    x = compose(
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)))(x)
    y = compose(
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D(out_filters, (1,1)))(x)
    return x, y


def yolo_body_1(inputs, num_anchors, num_classes):# 原版本模型
    """Create YOLO_V3 model CNN body in Keras."""
    darknet = Model(inputs, darknet_body(inputs))
    x, y1 = make_last_layers(darknet.output, 512, num_anchors*(num_classes+5))

    x = compose(
            DarknetConv2D_BN_Leaky(256, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x,darknet.layers[152].output])
    x, y2 = make_last_layers(x, 256, num_anchors*(num_classes+5))

    x = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x,darknet.layers[92].output])
    x, y3 = make_last_layers(x, 128, num_anchors*(num_classes+5))

    return Model(inputs, [y1,y2,y3])

def yolo_body(inputs, num_anchors, num_classes):### 高斯模型可配合1或2
    """Create YOLO_V3 model CNN body in Keras."""
    darknet = Model(inputs, darknet_body(inputs))
    x, y1 = make_last_layers(darknet.output, 512, num_anchors*(num_classes+9))###
    x = compose(DarknetConv2D_BN_Leaky(256, (1,1)),UpSampling2D(2))(x)
    x = Concatenate()([x,darknet.layers[152].output])
    x, y2 = make_last_layers(x, 256, num_anchors*(num_classes+9))###
    x = compose(DarknetConv2D_BN_Leaky(128, (1,1)),UpSampling2D(2))(x)
    x = Concatenate()([x,darknet.layers[92].output])
    x, y3 = make_last_layers(x, 128, num_anchors*(num_classes+9))###

    return Model(inputs, [y1,y2,y3])

def tiny_yolo_body(inputs, num_anchors, num_classes):
    '''Create Tiny YOLO_v3 model CNN body in keras.'''
    x1 = compose(
            DarknetConv2D_BN_Leaky(16, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(32, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(64, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(128, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(256, (3,3)))(inputs)
    x2 = compose(
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(512, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same'),
            DarknetConv2D_BN_Leaky(1024, (3,3)),
            DarknetConv2D_BN_Leaky(256, (1,1)))(x1)
    y1 = compose(
            DarknetConv2D_BN_Leaky(512, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x2)

    x2 = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x2)
    y2 = compose(
            Concatenate(),
            DarknetConv2D_BN_Leaky(256, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))([x2,x1])

    return Model(inputs, [y1,y2])


def yolo_head_1(feats, anchors, num_classes, input_shape, calc_loss=False):# 原函数
    """Convert final layer features to bounding box parameters."""
    # 将最终图层要素转换为边界框参数(yolo_outputs[l],anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.将矩阵型数据anchors，变为张量，并重塑形状
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    grid_shape = K.shape(feats)[1:3] # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
        [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
        [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    feats = K.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5]) ###

    # Adjust preditions to each spatial grid point and anchor size.
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs

def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):### 配合高斯版本2
    """Convert final layer features to bounding box parameters."""
    # (yolo_outputs[l], anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    grid_shape = K.shape(feats)[1:3] # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
        [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
        [grid_shape[0], 1, 1, 1])
    ### grid = K.concatenate([grid_x, grid_y]) # grid = K.concatenate([grid_x, grid_y], axis=-1)
    grid = K.concatenate([grid_x, grid_y], axis=-1)
    ### grid = K.cast(grid, K.dtype(feats))  # [?,?,?,42]

    feats = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 9])
    grid = K.cast(grid, K.dtype(feats))### 从上边换到下边
    # Adjust preditions to each spatial grid point and anchor size.
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))

    box_delta_xy = K.sigmoid(feats[..., :2])
    box_log_wh = feats[..., 2:4]
    box_sigma = K.sigmoid(feats[..., 4:8])
    box_confidence = K.sigmoid(feats[..., 8:9])
    box_class_probs = K.sigmoid(feats[..., 9:])

    if calc_loss == True:
        return grid, box_delta_xy, box_log_wh, box_sigma, box_xy, box_wh, box_confidence, box_class_probs
        # grid, raw_pred_delta_xy, raw_pred_log_wh, raw_pred_sigma, pred_xy, pred_wh, pred_confidence, pred_class
    return box_xy, box_wh, box_confidence, box_class_probs

def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    '''Get corrected boxes'''
    # 获取更正框
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    new_shape = K.round(image_shape * K.min(input_shape/image_shape))
    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes =  K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    '''Process Conv layer output'''
    # 处理转换层输出
    _, _, _,box_sigma,box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats,
        anchors, num_classes, input_shape,calc_loss=True)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4])
    # box_scores = box_confidence * box_class_probs # 原版
    ### 高斯版本
    box_scores = box_confidence * box_class_probs * (1 - tf.reduce_mean(box_sigma, axis=-1, keep_dims=True)) # 程序写法版本有些低，会引发报警
    ###
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores


def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              max_boxes=20,
              score_threshold=.6,
              iou_threshold=.5):
    """Evaluate YOLO model on given input and return filtered boxes."""
    # 在给定的输入和返回过滤框上评估YOLO模型。
    num_layers = len(yolo_outputs)
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]] # default setting
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
            anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        # TODO: use keras backend instead of tf.
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_

# 20200306：增加https://github.com/KUASWoodyLIN/keras-yolo3的改进型
def yolo_eval_v2(yolo_outputs_shape,
                 anchors,
                 num_classes,
                 image_shape,
                 max_boxes=20,
                 score_threshold=.6,
                 iou_threshold=.5):
    """Evaluate YOLO model on given input and return filtered boxes."""
    inputs = [K.placeholder(shape=(1, ) + shape[1:]) for shape in yolo_outputs_shape]
    num_layers = len(inputs)
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]] # default setting
    input_shape = K.shape(inputs[0])[1:3] * 32
    boxes = []
    box_scores = []
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(inputs[l],
            anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)
    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        # TODO: use keras backend instead of tf.
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_, inputs


def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    '''Preprocess true boxes to training input format
    为训练输入预处理真盒信息
    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5)
        Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
    input_shape: array-like, hw, multiples of 32
    anchors: array, shape=(N, 2), wh
    num_classes: integer

    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value

    '''
    assert (true_boxes[..., 4]<num_classes).all(), 'class id must be less than num_classes' # 类别id必须小于类别最大数
    num_layers = len(anchors)//3 # default setting
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]

    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')# （416x416）
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy/input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh/input_shape[::-1]

    m = true_boxes.shape[0] # 计算true_boxes里有几个值即一个batch中有几副图
    grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(num_layers)]
    y_true = [np.zeros((m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l]),5+num_classes),
        dtype='float32') for l in range(num_layers)]
    # y_true = [np.zeros((grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l]),5+num_classes),dtype='float32') for l in range(num_layers)]
    # y_true创建初始矩阵，以13框为例，y_true = (m=1，grid_shapes[L][0]=13,grid_shapes[L][1]=13,len(anchor_mask[L])=3,5+num_classes=10)
    # Expand dim to apply broadcasting.拓展1维
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0]>0 # bool码

    for b in range(m):
        # Discard zero rows. 放弃0行，box真实值与anchor之间联动 20200304
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh)==0: continue
        # Expand dim to apply broadcasting. 展开dim以应用广播。
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)   # 标靶与真实框最小相交
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)# 标靶与真实框最大相交
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)# 计算相交x,y
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]   # 计算相交区域
        box_area = wh[..., 0] * wh[..., 1]                             # 真实框区域
        anchor_area = anchors[..., 0] * anchors[..., 1]                # 标靶区域
        iou = intersect_area / (box_area + anchor_area - intersect_area) # 计算标靶区域占相交总区域的比重

        # Find best anchor for each true box 为每个真实框体找到最合适的anchor
        best_anchor = np.argmax(iou, axis=-1) # 计算9个标靶区域谁最大即最合适

        for t, n in enumerate(best_anchor): # enumerate：循环列举
            for l in range(num_layers):
                if n in anchor_mask[l]: # grid_shapes是啥？
                    i = np.floor(true_boxes[b,t,0]*grid_shapes[l][1]).astype('int32') # 真实框中的x/416形式数*
                    j = np.floor(true_boxes[b,t,1]*grid_shapes[l][0]).astype('int32') #
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b,t, 4].astype('int32') # 提取类别号
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b,t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5+c] = 1

    return y_true


def box_iou(b1, b2):
    '''Return iou tensor

    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh

    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)

    '''

    # Expand dim to apply broadcasting.
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


def yolo_loss_1(args, anchors, num_classes, ignore_thresh=.5, print_loss=False):
    '''Return yolo_loss tensor 原loss函数

    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true: list of array, the output of preprocess_true_boxes
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss

    Returns
    -------
    loss: tensor, shape=(1,)

    '''
    num_layers = len(anchors)//3 # default setting
    yolo_outputs = args[:num_layers]
    y_true = args[num_layers:]
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]
    loss = 0
    m = K.shape(yolo_outputs[0])[0] # batch size, tensor
    mf = K.cast(m, K.dtype(yolo_outputs[0]))

    for l in range(num_layers):
        object_mask = y_true[l][..., 4:5]
        true_class_probs = y_true[l][..., 5:] # y_true_class_probs
        # grid, y_pred_box, y_pred_delta_xy, y_pred_log_wh, y_pred_sigma, y_pred_confidence, y_pred_class_probs = y_pred_graph(raw_y_pred, anchors[anchor_mask], input_shape)
        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l],
             anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)
        pred_box = K.concatenate([pred_xy, pred_wh])

        # Darknet raw box to calculate loss. 计算损失
        raw_true_xy = y_true[l][..., :2]*grid_shapes[l][::-1] - grid
        raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])
        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh)) # avoid log(0)=-inf
        box_loss_scale = 2 - y_true[l][...,2:3]*y_true[l][...,3:4]

        # Find ignore mask, iterate over each of batch.
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')
        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[l][b,...,0:4], object_mask_bool[b,...,0])
            iou = box_iou(pred_box[b], true_box)
            best_iou = K.max(iou, axis=-1)
            ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_thresh, K.dtype(true_box)))
            return b+1, ignore_mask
        _, ignore_mask = K.control_flow_ops.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = K.expand_dims(ignore_mask, -1)

        # # K.binary_crossentropy is helpful to avoid exp overflow.二元交叉熵有助于避免exp溢出。
        xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[...,0:2], from_logits=True)
        wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh-raw_pred[...,2:4])
        confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True)+ \
            (1-object_mask) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) * ignore_mask
        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[...,5:], from_logits=True)

        xy_loss = K.sum(xy_loss) / mf
        wh_loss = K.sum(wh_loss) / mf
        confidence_loss = K.sum(confidence_loss) / mf
        class_loss = K.sum(class_loss) / mf
        loss += xy_loss + wh_loss + confidence_loss + class_loss

        ### 高斯loss
        # y_true = tf.concat([raw_true_xy, raw_true_wh], axis=-1)
        # y_pred_mu = tf.concat([y_pred_delta_xy, y_pred_log_wh], axis=-1)
        ###
        if print_loss:
            loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message='loss: ')
    return loss

def yolo_loss_g(args, anchors, num_classes, ignore_thresh=.5, print_loss=False):
    '''Return yolo_loss tensor
    高斯版本1
    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true: list of array, the output of preprocess_true_boxes
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss

    Returns
    -------
    loss: tensor, shape=(1,)

    '''
    num_anchors_per_layer = 3
    num_output_layers = len(anchors) // num_anchors_per_layer
    yolo_outputs = args[:num_output_layers]
    raw_y_trues = args[num_output_layers:]
    anchor_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(raw_y_trues[0]))
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(raw_y_trues[0])) for l in range(num_output_layers)]
    loss = 0
    batch_size = K.shape(yolo_outputs[0])[0]
    batch_size_f = K.cast(batch_size, K.dtype(yolo_outputs[0]))

    for l in range(num_output_layers):
        # grid_shape = grid_shapes[l]
        yolo_output = yolo_outputs[l]
        grid_shape = K.shape(yolo_output)[1:3]
        raw_y_pred = K.reshape(yolo_output, [-1, grid_shape[0], grid_shape[1], num_anchors_per_layer, num_classes + 9]) ####
        # raw_y_pred = K.reshape(yolo_output, (-1, grid_shape[0], grid_shape[1], 3, num_classes + 9))
        # raw_y_pred = K.reshape(yolo_output, [-1, -1, -1, 3, 14])
        raw_y_true = raw_y_trues[l]
        anchor_mask = anchor_masks[l]
        # (batch_size, grid_height, grid_width, num_anchors_this_layer, 1)
        object_mask = raw_y_true[..., 4:5]
        # (batch_size, grid_height, grid_width, num_anchors_this_layer, num_classes)
        y_true_class_probs = raw_y_true[..., 5:]
        grid, y_pred_box, y_pred_delta_xy, y_pred_log_wh, y_pred_sigma, y_pred_confidence, y_pred_class_probs = \
            y_pred_graph(raw_y_pred, anchors[anchor_mask], input_shape)
        y_true_delta_xy = raw_y_true[..., :2] * grid_shapes[l][::-1] - grid
        y_true_log_wh = K.log(raw_y_true[..., 2:4] * input_shape[::-1] / anchors[anchor_mask])
        y_true_log_wh = K.switch(object_mask, y_true_log_wh, K.zeros_like(y_true_log_wh))  # raw_true_wh
        box_loss_scale = 2 - raw_y_true[..., 2:3] * raw_y_true[..., 3:4]
        ignore_mask = tf.TensorArray(K.dtype(raw_y_trues[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')

        def loop_body(b, ignore_mask_):
            # (num_gt_boxes, 4)
            gt_box = tf.boolean_mask(raw_y_true[b, ..., 0:4], object_mask_bool[b, ..., 0])
            # (grid_height, grid_width, num_anchors_this_layer, num_gt_boxes)
            iou = box_iou_graph(y_pred_box[b], gt_box)
            # (grid_height, grid_width, num_anchors_this_layer)
            best_iou = K.max(iou, axis=-1)
            ignore_mask_ = ignore_mask_.write(b, K.cast(best_iou < ignore_thresh, K.dtype(gt_box)))
            return b + 1, ignore_mask_

        _, ignore_mask = tf.while_loop(lambda b, *largs: b < batch_size, loop_body, [0, ignore_mask])
        # (batch_size, grid_height, grid_width, num_anchors_this_layer)
        ignore_mask = ignore_mask.stack()
        # (batch_size, grid_height, grid_width, num_anchors_this_layer, 1)
        ignore_mask = K.expand_dims(ignore_mask, -1)

        y_true = tf.concat([y_true_delta_xy, y_true_log_wh], axis=-1)
        y_pred_mu = tf.concat([y_pred_delta_xy, y_pred_log_wh], axis=-1)
        x_loss = nll_loss(y_true[..., 0:1], y_pred_mu[..., 0:1], y_pred_sigma[..., 0:1])
        x_loss = object_mask * box_loss_scale * x_loss
        y_loss = nll_loss(y_true[..., 1:2], y_pred_mu[..., 1:2], y_pred_sigma[..., 1:2])
        y_loss = object_mask * box_loss_scale * y_loss
        w_loss = nll_loss(y_true[..., 2:3], y_pred_mu[..., 2:3], y_pred_sigma[..., 2:3])
        w_loss = object_mask * box_loss_scale * w_loss
        h_loss = nll_loss(y_true[..., 3:4], y_pred_mu[..., 3:4], y_pred_sigma[..., 3:4])
        h_loss = object_mask * box_loss_scale * h_loss
        confidence_loss = object_mask * K.binary_crossentropy(object_mask, y_pred_confidence) + \
                          (1 - object_mask) * K.binary_crossentropy(object_mask, y_pred_confidence) * ignore_mask
        class_loss = object_mask * K.binary_crossentropy(y_true_class_probs, y_pred_class_probs)
        x_loss = K.sum(x_loss) / batch_size_f
        y_loss = K.sum(y_loss) / batch_size_f
        w_loss = K.sum(w_loss) / batch_size_f
        h_loss = K.sum(h_loss) / batch_size_f
        confidence_loss = K.sum(confidence_loss) / batch_size_f
        class_loss = K.sum(class_loss) / batch_size_f
        loss += x_loss + y_loss + w_loss + h_loss + confidence_loss + class_loss
        if print_loss:
            loss = tf.Print(loss,
                            [loss, x_loss, y_loss, w_loss, h_loss, confidence_loss, class_loss, K.sum(ignore_mask)],
                            message='\nloss: ')
    return loss


def yolo_loss(args, anchors, num_classes, ignore_thresh=.5, print_loss=False):
    '''Return yolo_loss tensor
    高斯版本2
    ----------
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true: list of array, the output of preprocess_true_boxes
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss
    Returns
    -------
    loss: tensor, shape=(1,)
    '''
    num_layers = len(anchors)//3 # default setting
    yolo_outputs = args[:num_layers]
    y_true = args[num_layers:]
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]
    loss = 0
    m = K.shape(yolo_outputs[0])[0] # batch size, tensor
    batch_size_f = K.cast(m, K.dtype(yolo_outputs[0]))

    for l in range(num_layers):
        object_mask = y_true[l][..., 4:5]
        true_class_probs = y_true[l][..., 5:]

        # grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l],
        #      anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)
        grid, raw_pred_delta_xy, raw_pred_log_wh, raw_pred_sigma, pred_xy, pred_wh, pred_confidence, pred_class = yolo_head(yolo_outputs[l],
                                                     anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)
        pred_box = K.concatenate([pred_xy, pred_wh])

        # Darknet raw box to calculate loss.
        raw_true_xy = y_true[l][..., :2]*grid_shapes[l][::-1] - grid
        raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])
        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh)) # avoid log(0)=-inf
        box_loss_scale = 2 - y_true[l][...,2:3]*y_true[l][...,3:4]

        # Find ignore mask, iterate over each of batch.
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')
        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[l][b,...,0:4], object_mask_bool[b,...,0])
            iou = box_iou(pred_box[b], true_box)
            best_iou = K.max(iou, axis=-1)
            ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_thresh, K.dtype(true_box)))
            return b+1, ignore_mask
        _, ignore_mask = K.control_flow_ops.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = K.expand_dims(ignore_mask, -1)

        # K.binary_crossentropy is helpful to avoid exp overflow.
        # xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[...,0:2], from_logits=True)
        # wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh-raw_pred[...,2:4])
        # confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True)+ \
        #     (1-object_mask) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) * ignore_mask
        # class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[...,5:], from_logits=True)
        #
        # xy_loss = K.sum(xy_loss) / mf
        # wh_loss = K.sum(wh_loss) / mf
        # confidence_loss = K.sum(confidence_loss) / mf
        # class_loss = K.sum(class_loss) / mf
        raw_true = tf.concat([raw_true_xy, raw_true_wh], axis=-1)
        y_pred_mu = tf.concat([raw_pred_delta_xy, raw_pred_log_wh], axis=-1)

        x_loss = nll_loss(raw_true[..., 0:1], y_pred_mu[..., 0:1], raw_pred_sigma[..., 0:1])
        x_loss = object_mask * box_loss_scale * x_loss

        y_loss = nll_loss(raw_true[..., 1:2], y_pred_mu[..., 1:2], raw_pred_sigma[..., 1:2])
        y_loss = object_mask * box_loss_scale * y_loss
        w_loss = nll_loss(raw_true[..., 2:3], y_pred_mu[..., 2:3], raw_pred_sigma[..., 2:3])
        w_loss = object_mask * box_loss_scale * w_loss
        h_loss = nll_loss(raw_true[..., 3:4], y_pred_mu[..., 3:4], raw_pred_sigma[..., 3:4])
        h_loss = object_mask * box_loss_scale * h_loss

        # confidence_loss = object_mask * K.binary_crossentropy(object_mask, pred_confidence, from_logits=True)+ \
        #      (1-object_mask) * K.binary_crossentropy(object_mask, pred_confidence, from_logits=True) * ignore_mask
        # class_loss = object_mask * K.binary_crossentropy(true_class_probs, pred_class, from_logits=True)

        confidence_loss = object_mask * K.binary_crossentropy(object_mask, pred_confidence)+ \
             (1-object_mask) * K.binary_crossentropy(object_mask, pred_confidence) * ignore_mask
        class_loss = object_mask * K.binary_crossentropy(true_class_probs, pred_class)

        x_loss = K.sum(x_loss) / batch_size_f
        y_loss = K.sum(y_loss) / batch_size_f
        w_loss = K.sum(w_loss) / batch_size_f
        h_loss = K.sum(h_loss) / batch_size_f

        confidence_loss = K.sum(confidence_loss) / batch_size_f
        class_loss = K.sum(class_loss) / batch_size_f
        loss += x_loss + y_loss + w_loss + h_loss +  confidence_loss + class_loss
        if print_loss:
            loss = tf.Print(loss, [loss, x_loss,y_loss, w_loss,h_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message='loss: ')
    return loss