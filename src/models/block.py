#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf

__date__ = '2020/06/09'


class FilterResponseNormalization(tf.keras.layers.Layer):
    def __init__(self):
        super(FilterResponseNormalization, self).__init__()

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        data_format = tf.keras.backend.image_data_format()
        axis = 1 if data_format == 'channels_first' else -1

        self.gamma = self.add_weight(
            'gamma', shape=[input_shape[axis]], dtype=tf.float32,
            initializer=tf.keras.initializers.ones(), trainable=True
        )
        self.beta = self.add_weight(
            'beta', shape=[input_shape[axis]], dtype=tf.float32,
            initializer=tf.keras.initializers.zeros(), trainable=True
        )
        self.tau = self.add_weight(
            'tau', shape=[input_shape[axis]], dtype=tf.float32,
            initializer=tf.keras.initializers.zeros(), trainable=True
        )
        self.epsilon = self.add_weight(
            'epsilon', shape=[input_shape[axis]], dtype=tf.float32,
            initializer=tf.keras.initializers.zeros(), trainable=True
        )

        self.axis = (2, 3) if data_format == 'channels_first' else (1, 2)
        self.built = True

    def call(self, inputs, **kwargs):
        nu_squared = tf.math.reduce_sum(
            tf.math.square(inputs), axis=self.axis, keepdims=True
        )
        epsilon = 1e-6 + tf.nn.softplus(self.epsilon)
        x = inputs * tf.math.rsqrt(nu_squared + epsilon)

        return tf.math.maximum(self.gamma * x + self.beta, self.tau)


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters, strides=1, use_se=True,
                 trainable=True, name=None, dtype=None, dynamic=False,
                 **kwargs):
        super(ResidualBlock, self).__init__(
            trainable=trainable, name=name, dtype=dtype, dynamic=dynamic,
            **kwargs
        )
        self.filters = filters
        self.strides = strides
        self.use_se = use_se

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        data_format = tf.keras.backend.image_data_format()
        axis = 1 if data_format == 'channels_first' else -1

        self.residual = tf.keras.models.Sequential([
            FilterResponseNormalization(),
            tf.keras.layers.Conv2D(filters=input_shape[axis], kernel_size=3,
                                   strides=self.strides, padding='same',
                                   use_bias=True),
            FilterResponseNormalization(),
            tf.keras.layers.Conv2D(filters=self.filters, kernel_size=3,
                                   padding='same')
        ])
        if self.use_se:
            self.residual.add(SqueezeExcitation())

        if self.filters != input_shape[0]:
            self.bypath = tf.keras.layers.Conv2D(
                filters=self.filters, kernel_size=1, strides=self.strides,
                padding='same'
            )
        elif self.strides != 1:
            self.bypath = tf.keras.layers.Conv2D(
                filters=self.filters, kernel_size=3, strides=self.strides,
                padding='same'
            )
        else:
            self.bypath = lambda inputs: inputs

        self.built = True

    def call(self, inputs, training=None, mask=None):
        h = self.residual(inputs=inputs, training=training)
        x = self.bypath(inputs=inputs)
        return x + h

    def get_config(self):
        d = {'filters': self.filters, 'strides': self.strides,
             'use_se': self.use_se}
        d.update(super(ResidualBlock, self).get_config())
        return d


class SqueezeExcitation(tf.keras.models.Model):
    def __init__(self):
        super(SqueezeExcitation, self).__init__()

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        data_format = tf.keras.backend.image_data_format()
        axis = 1 if data_format == 'channels_first' else -1

        shape = ((input_shape[axis], 1, 1) if data_format == 'channels_first'
                 else (1, 1, input_shape[axis]))

        self.c_se = tf.keras.layers.Conv2D(
            filters=1, kernel_size=1, activation='sigmoid'
        )

        self.s_se = tf.keras.models.Sequential([
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(units=input_shape[axis] // 2,
                                  activation='relu'),
            tf.keras.layers.Dense(units=input_shape[axis],
                                  activation='sigmoid'),
            tf.keras.layers.Reshape(target_shape=shape)
        ])

        self.built = True

    def call(self, inputs, training=None, mask=None):
        hc = self.c_se(inputs)
        hs = self.s_se(inputs)

        outputs = hc * inputs + hs * inputs
        return outputs

    def get_config(self):
        return super(SqueezeExcitation, self).get_config()


class PyramidalBottleneckBlock(tf.keras.Model):
    """
    https://arxiv.org/pdf/1610.02915.pdf
    Fig.6(d)
    """
    def __init__(self, filters, strides=1):
        super(PyramidalBottleneckBlock, self).__init__()

        self.filters = filters
        self.strides = strides

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        data_format = tf.keras.backend.image_data_format()
        axis = 1 if data_format == 'channels_first' else -1

        self.net = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(
                axis=axis, center=True, scale=False,
                input_shape=input_shape[1:]
            ),
            tf.keras.layers.Conv2D(
                filters=input_shape[axis] // 4, kernel_size=1, padding='same',
                use_bias=False
            ),
            tf.keras.layers.BatchNormalization(
                axis=axis, center=True, scale=False
            ),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(
                filters=input_shape[axis] // 4, kernel_size=3, padding='same',
                use_bias=False, strides=self.strides
            ),
            tf.keras.layers.BatchNormalization(
                axis=axis, center=True, scale=False
            ),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(
                filters=self.filters, kernel_size=1, padding='same',
                use_bias=False
            ),
            tf.keras.layers.BatchNormalization(
                axis=axis, center=True, scale=True
            )
        ])

        if self.strides != 1:
            self.identity = tf.keras.layers.SeparableConv2D(
                filters=input_shape[axis], kernel_size=3, strides=self.strides,
                padding='same', input_shape=input_shape
            )
        else:
            self.identity = lambda inputs: inputs
        self.concat = tf.keras.layers.Concatenate(axis=axis)

        output_shape = self.net.output_shape
        if data_format == 'channels_first':
            pad_size = (self.filters - input_shape[1],) + output_shape[2:]
        else:
            pad_size = output_shape[1:3] + (self.filters - input_shape[3],)
        pad_size = (1,) + pad_size
        self.pad = tf.zeros(pad_size)

        self.add = tf.keras.layers.Add()

    def call(self, inputs, training=None, mask=None):
        i = self.identity(inputs)

        batch_size = tf.shape(inputs)[0]
        zero = tf.tile(self.pad, [batch_size, 1, 1, 1])
        x = self.concat([i, zero])

        h = self.net(inputs, training=training)

        outputs = self.add([x, h])

        return outputs


class InceptionBlock(tf.keras.layers.Layer):
    def __init__(self, filters, trainable=True, name=None, dtype=None,
                 dynamic=False, **kwargs):
        super(InceptionBlock, self).__init__(
            trainable=trainable, name=name, dtype=dtype, dynamic=dynamic,
            **kwargs
        )
        self.filters = filters

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        self.branch1 = tf.keras.layers.Conv2D(
            filters=self.filters // 4, kernel_size=1, padding='same'
        )
        self.branch2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                filters=self.filters // 4, kernel_size=1, padding='same'
            ),
            tf.keras.layers.Conv2D(
                filters=self.filters // 4, kernel_size=3, padding='same'
            )
        ])
        self.branch3 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(
                filters=self.filters // 4, kernel_size=1, padding='same'
            ),
            tf.keras.layers.Conv2D(
                filters=self.filters // 4, kernel_size=3, padding='same'
            ),
            tf.keras.layers.Conv2D(
                filters=self.filters // 4, kernel_size=3, padding='same'
            )
        ])
        self.branch4 = tf.keras.Sequential([
            tf.keras.layers.AveragePooling2D(
                pool_size=3, strides=1, padding='same'
            ),
            tf.keras.layers.Conv2D(
                filters=self.filters // 4, kernel_size=1, padding='same'
            )
        ])

        self.concat = tf.keras.layers.Concatenate()

        self.built = True

    def call(self, inputs, **kwargs):
        branch1 = self.branch1(inputs, **kwargs)
        branch2 = self.branch2(inputs, **kwargs)
        branch3 = self.branch3(inputs, **kwargs)
        branch4 = self.branch4(inputs, **kwargs)

        outputs = self.concat([branch1, branch2, branch3, branch4])
        return outputs


class ResBlock(tf.keras.layers.Layer):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False,
                 **kwargs):
        super(ResBlock, self).__init__(
            trainable=trainable, name=name, dtype=dtype, dynamic=dynamic,
            **kwargs
        )

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        data_format = tf.keras.backend.image_data_format()
        axis = 1 if data_format == 'channels_first' else -1

        filters = input_shape[axis]
        self.residual = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(axis=axis),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=filters, kernel_size=3,
                                   padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(axis=axis),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=filters, kernel_size=3,
                                   padding='same')
        ])

        self.built = True

    def call(self, inputs, training=None, mask=None):
        h = self.residual(inputs, training=training)
        outputs = h + inputs
        return outputs
