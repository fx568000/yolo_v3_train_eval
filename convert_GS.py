
import io
import os
import argparse
import configparser
import numpy as np
import tensorflow as tf
from collections import defaultdict

from keras import backend as K
from keras.layers import (Conv2D, Input, ZeroPadding2D, Add,UpSampling2D, MaxPooling2D, Concatenate)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from keras.utils.vis_utils import plot_model as plot

tf.app.flags.DEFINE_string('config_path','Gaussian_yolov3_BDD.cfg','Path to Darknet cfg file.')
tf.app.flags.DEFINE_string('weights_path','Gaussian_yolov3_BDD.weights','Path to Darknet weights file.') # yolov3
tf.app.flags.DEFINE_string('output_path','model_data/yolo_g.h5','Path to output Keras model file.')
FLAGS = tf.app.flags.FLAGS

parser = argparse.ArgumentParser(description='Darknet To Keras Converter.')
parser.add_argument('-p','--plot_model',help='Plot generated Keras model and save as image.',action='store_true')
parser.add_argument('-w','--weights_only',help='Save as Keras weights file instead of model file.',action='store_true')

def unique_config_sections(config_file):
    """
    Convert all config sections to have unique names.
    Adds unique suffixes to config sections for compatibility with configparser.
    Args:
        config_file:
    Returns:
    """
    section_counters = defaultdict(int)
    output_stream = io.StringIO()
    with open(config_file) as fin:
        for line in fin:
            if line.startswith('['):
                section = line.strip().strip('[]')
                _section = section + '_' + str(section_counters[section])
                section_counters[section] += 1
                line = line.replace(section, _section)
            output_stream.write(line)
    output_stream.seek(0)
    return output_stream


def main(args):
    config_path = os.path.expanduser(FLAGS.config_path)
    weights_path = os.path.expanduser(FLAGS.weights_path)
    output_path = os.path.expanduser(FLAGS.output_path)
    assert config_path.endswith('.cfg'), 'config path {} is not a .cfg file'.format(config_path)
    assert weights_path.endswith('.weights'), 'weights path {} is not a .weights file'.format(weights_path)
    assert output_path.endswith('.h5'), 'output path {} is not a .h5 file'.format(output_path)

    # Load weights and config.
    print('Loading weights.')
    weights_file = open(weights_path, 'rb')
    major, minor, revision = np.ndarray(shape=(3,), dtype='int32', buffer=weights_file.read(12))
    if (major * 10 + minor) >= 2 and major < 1000 and minor < 1000:
        seen = np.ndarray(shape=(1,), dtype='int64', buffer=weights_file.read(8))
    else:
        seen = np.ndarray(shape=(1,), dtype='int32', buffer=weights_file.read(4))
    print('Weights Header: ', major, minor, revision, seen)

    print('Parsing Darknet config.')
    unique_config_file = unique_config_sections(config_path)
    cfg_parser = configparser.ConfigParser()
    cfg_parser.read_file(unique_config_file)

    print('Creating Keras model.')
    input_layer = Input(shape=(416, 416, 3))
    prev_layer = input_layer
    all_layers = []

    weight_decay = float(cfg_parser['net_0']['decay']) if 'net_0' in cfg_parser.sections() else 5e-4
    four_bytes_consumed_count = 0
    out_index = []
    for section in cfg_parser.sections():
        print('Parsing section {}'.format(section))
        if section.startswith('convolutional'):
            filters = int(cfg_parser[section]['filters'])
            size = int(cfg_parser[section]['size'])
            stride = int(cfg_parser[section]['stride'])
            pad = int(cfg_parser[section]['pad'])
            activation = cfg_parser[section]['activation']
            batch_normalize = 'batch_normalize' in cfg_parser[section]

            padding = 'same' if pad == 1 and stride == 1 else 'valid'

            # Setting weights.
            # Darknet serializes convolutional weights as:
            # [bias/beta, [gamma, mean, variance], conv_weights]
            prev_layer_shape = K.int_shape(prev_layer)
            # Note 每一个 filter 的 shape 为 (size, size, pre_layer_shape[-1])
            # (outdim, indim, height, width)
            weights_shape = (filters, prev_layer_shape[-1], size, size)

            weights_size = np.product(weights_shape)
            print('conv2d', 'bn' if batch_normalize else '  ', activation, weights_shape)

            conv_bias = np.ndarray(
                shape=(filters,),
                dtype='float32',
                buffer=weights_file.read(filters * 4))
            four_bytes_consumed_count += filters
            if batch_normalize:
                bn_weights = np.ndarray(
                    shape=(3, filters),
                    dtype='float32',
                    buffer=weights_file.read(filters * 12))
                four_bytes_consumed_count += 3 * filters

                bn_weight_list = [
                    # scale gamma
                    bn_weights[0],
                    # shift beta
                    conv_bias,
                    # running mean
                    bn_weights[1],
                    # running var
                    bn_weights[2]
                ]
            conv_weights = np.ndarray(
                shape=weights_shape, ###
                dtype='float32',
                buffer=weights_file.read(weights_size * 4))
            four_bytes_consumed_count += weights_size

            # DarkNet conv_weights are serialized Caffe-style:
            # (out_dim, in_dim, height, width)
            # We would like to set these to Tensorflow order:
            # (height, width, in_dim, out_dim)
            conv_weights = np.transpose(conv_weights, [2, 3, 1, 0])
            conv_weights = [conv_weights] if batch_normalize else [conv_weights, conv_bias]

            # Handle activation.
            if activation not in ('leaky', 'linear'):
                raise ValueError('Unknown activation function `{}` in section {}'.format(activation, section))

            # Create Conv2D layer
            if stride > 1:
                # Darknet uses left and top padding instead of 'same' mode when stride > 1
                prev_layer = ZeroPadding2D(((1, 0), (1, 0)))(prev_layer)
            conv_layer = Conv2D(
                filters, (size, size),
                strides=(stride, stride),
                kernel_regularizer=l2(weight_decay),
                use_bias=not batch_normalize,
                weights=conv_weights,
                padding=padding)(prev_layer)

            if batch_normalize:
                conv_layer = (BatchNormalization(weights=bn_weight_list))(conv_layer)
            prev_layer = conv_layer

            if activation == 'linear':
                all_layers.append(prev_layer)
            elif activation == 'leaky':
                act_layer = LeakyReLU(alpha=0.1)(prev_layer)
                prev_layer = act_layer
                all_layers.append(act_layer)

        elif section.startswith('route'):
            # concatenate layers
            ids = [int(i) for i in cfg_parser[section]['layers'].split(',')]
            layers = [all_layers[i] for i in ids]
            if len(layers) > 1:
                print('Concatenating route layers:', layers)
                concatenate_layer = Concatenate()(layers)
                all_layers.append(concatenate_layer)
                prev_layer = concatenate_layer
            else:
                # only one layer to route
                skip_layer = layers[0]
                all_layers.append(skip_layer)
                prev_layer = skip_layer

        elif section.startswith('maxpool'):
            size = int(cfg_parser[section]['size'])
            stride = int(cfg_parser[section]['stride'])
            all_layers.append(
                MaxPooling2D(
                    pool_size=(size, size),
                    strides=(stride, stride),
                    padding='same')(prev_layer))
            prev_layer = all_layers[-1]

        elif section.startswith('shortcut'):
            index = int(cfg_parser[section]['from'])
            activation = cfg_parser[section]['activation']
            assert activation == 'linear', 'Only linear activation supported.'
            all_layers.append(Add()([all_layers[index], prev_layer]))
            prev_layer = all_layers[-1]

        elif section.startswith('upsample'):
            stride = int(cfg_parser[section]['stride'])
            assert stride == 2, 'Only stride=2 supported.'
            all_layers.append(UpSampling2D(stride)(prev_layer))
            prev_layer = all_layers[-1]

        elif section.startswith('yolo'):
            out_index.append(len(all_layers) - 1)
            all_layers.append(None)
            prev_layer = all_layers[-1]

        elif section.startswith('net'):
            pass

        else:
            raise ValueError('Unsupported section header type: {}'.format(section))

    # Create and save model.
    if len(out_index) == 0:
        out_index.append(len(all_layers) - 1)
    model = Model(inputs=input_layer, outputs=[all_layers[i] for i in out_index])

    if args.weights_only:
        model.save_weights('{}'.format(output_path))
        print('Saved Keras weights to {}'.format(output_path))
    else:
        model.save('{}'.format(output_path))
        print('Saved Keras model to {}'.format(output_path))

    # Check to see if all weights have been read.
    remaining_weights = len(weights_file.read()) // 4
    weights_file.close()
    print('Read {} of {} from Darknet weights.'.format(four_bytes_consumed_count,
                                                       four_bytes_consumed_count + remaining_weights))
    if remaining_weights > 0:
        print('Warning: {} unused weights'.format(remaining_weights))

    if args.plot_model:
        output_root = os.path.splitext(output_path)[0]
        plot(model, to_file='{}.png'.format(output_root), show_shapes=True)
        print('Saved model plot to {}.png'.format(output_root))

if __name__ == '__main__':
    main(parser.parse_args())