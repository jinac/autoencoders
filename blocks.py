"""
Custom Layer Blocks.
"""
from tensorflow.keras import backend
from tensorflow.keras.layers import (Activation,
                                     BatchNormalization,
                                     Conv2D,
                                     Conv2DTranspose,Dense,
                                     Dropout,
                                     LeakyReLU,
                                     Reshape,
                                     SeparableConv2D)

import math

def bn_dense(inputs, out_dim,
             activation='relu',
             batchnorm=True,
             dropout_rate=None,
             use_bias=True):
    out = Dense(out_dim, use_bias=use_bias)(inputs)
    if batchnorm:
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
                  dropout_rate=None,
                  use_bias=True):
    out = Conv2D(num_filters, kernel_size,
                 strides=strides, padding=padding,
                 use_bias=use_bias)(inputs)
    if batchnorm:
        out = BatchNormalization()(out)
    if activation is not None:
        out = Activation(activation)(out)
    return out


def bn_sepconv_layer(inputs, num_filters, kernel_size,
                     strides=[1, 1], padding='same',
                     activation='relu',
                     batchnorm=True,
                     dropout_rate=None,
                     use_bias=True):
    out = SeparableConv2D(num_filters, kernel_size,
                          strides=strides, padding=padding,
                          use_bias=use_bias)(inputs)
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
                    batchnorm=True,
                    use_bias=True):
    out = Conv2DTranspose(num_filters,
                          kernel_size=kernel_size,
                          strides=strides,
                          padding=padding,
                          use_bias=use_bias)(inputs)
    if batchnorm:
        out = BatchNormalization()(out)
    if activation is not None:
        out = Activation(activation)(out)
    if dropout_rate is not None:
        out = Dropout(dropout_rate)(out)
    return out


def convnet(input_img,
            filter_param=64, max_filter=256,
            bias=True, batchnorm=True,
            activation='relu'):
    """
    """
    img_dim = input_img.shape[1:]
    num_layers = math.frexp(int(img_dim[0]))[1] - 4

    out = bn_conv_layer(input_img, int(img_dim[-1]), 4, 2,
                        batchnorm=batchnorm, activation=activation,
                        use_bias=bias)

    for n in range(num_layers):
        num_filters = min(filter_param * (2 ** n), max_filter)
        out = bn_conv_layer(out, num_filters, 4, 2,
                            batchnorm=batchnorm, activation=activation,
                            use_bias=bias)

    return out


def deconvnet(input_dense, output_dim,
              filter_param=16, min_filter=32,
              bias=True, batchnorm=True,
              activation='relu'):
    """
    """
    num_layers = math.frexp(output_dim[0])[1] - 4

    new_shape = (1, 1, int(input_dense.shape[-1]))
    out = Reshape(new_shape)(input_dense)
    out = bn_deconv_layer(out, filter_param * (2 ** (num_layers + 1)),
                          4, 4, batchnorm=batchnorm, activation=activation,
                          use_bias=bias)

    for n in range(num_layers):
        num_filters = max(filter_param * (2 ** (num_layers - n)), min_filter)
        out = bn_deconv_layer(out, num_filters, 4, 2,
                              batchnorm=batchnorm, activation=activation,
                              use_bias=bias)

    return out
