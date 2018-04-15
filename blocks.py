"""
Custom Layer Blocks.
"""
from keras import backend
from keras.layers import (Activation, Dense, Dropout, Reshape)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import (Conv2D,
                                        Conv2DTranspose,
                                        SeparableConv2D)
from keras.layers.normalization import BatchNormalization

import math

def bn_dense(inputs, out_dim,
             activation='relu',
             dropout_rate=None,
             use_bias=True):
    out = Dense(out_dim, use_bias=use_bias)(inputs)
    out = BatchNormalization()(out)
    if activation is not None:
        out = Activation(activation)(out)
    if dropout_rate is not None:
        out = Dropout(dropout_rate)(out)
    return out


def bn_conv_layer(inputs, num_filters, kernel_size,
                  strides=[1, 1], padding='same',
                  activation='relu',
                  batchnorm=True,
                  dropout_rate=None):
    out = Conv2D(num_filters, kernel_size,
                 strides=strides, padding=padding,
                 use_bias=True)(inputs)
    if batchnorm:
        out = BatchNormalization()(out)
    if activation is not None:
        out = Activation(activation)(out)
    return out


def bn_sepconv_layer(inputs, num_filters, kernel_size,
                  strides=[1, 1], padding='same',
                  activation='relu',
                  batchnorm=True,
                  dropout_rate=None):
    out = SeparableConv2D(num_filters, kernel_size,
                          strides=strides, padding=padding,
                          use_bias=True)(inputs)
    if batchnorm:
        out = BatchNormalization()(out)
    if activation is not None:
        out = Activation(activation)(out)
    return out


def bn_deconv_layer(inputs,
                    num_filters,
                    kernel_size,
                    strides,
                    padding='same',
                    activation='relu',
                    dropout_rate=None,
                    batchnorm=True):
    out = Conv2DTranspose(num_filters,
                          kernel_size=kernel_size,
                          strides=strides,
                          padding=padding,
                          use_bias=True)(inputs)
    if batchnorm:
        out = BatchNormalization()(out)
    if activation is not None:
        out = Activation(activation)(out)
    if dropout_rate is not None:
        out = Dropout(dropout_rate)(out)
    return out


def convnet(input_img, filter_param=64, max_filter=256):
    """
    """
    img_dim = input_img.shape[1:]
    num_layers = math.frexp(int(img_dim[0]))[1] - 4

    out = bn_conv_layer(input_img, int(img_dim[-1]), 4, 2)

    for n in xrange(num_layers):
        num_filters = min(filter_param * (2 ** n), max_filter)
        out = bn_conv_layer(out, num_filters, 4, 2)

    return out


def deconvnet(input_dense, output_dim, filter_param=16, min_filter=32):
    """
    """
    num_layers = math.frexp(output_dim[0])[1] - 4

    new_shape = (1, 1, int(input_dense.shape[-1]))
    out = Reshape(new_shape)(input_dense)
    out = bn_deconv_layer(out, filter_param * (2 ** (num_layers + 1)), 4, 4)

    for n in xrange(num_layers):
        num_filters = max(filter_param * (2 ** (num_layers - n)), min_filter)
        out = bn_deconv_layer(out, num_filters, 4, 2)

    return out
