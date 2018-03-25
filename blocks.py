"""
Custom Layer Blocks.
"""
from keras.layers import (Activation, Dense, Dropout)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import (Conv2D,
                                        Conv2DTranspose)
from keras.layers.normalization import BatchNormalization


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
                 use_bias=False)(inputs)
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
                          use_bias=False)(inputs)
    if batchnorm:
      out = BatchNormalization()(out)
    if activation is not None:
        out = Activation(activation)(out)
    if dropout_rate is not None:
        out = Dropout(dropout_rate)(out)
    return out
