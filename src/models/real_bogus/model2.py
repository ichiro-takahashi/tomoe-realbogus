#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reorganize codes of the classification model.
"""
import numpy as np
import tensorflow as tf

from models.block import ResidualBlock, ResBlock, InceptionBlock

__date__ = '2021/10/28'


class ExpHingeLoss(tf.keras.losses.Loss):
    def __init__(self, margin, reduction=tf.keras.losses.Reduction.AUTO,
                 name=None):
        super(ExpHingeLoss, self).__init__(reduction=reduction, name=name)
        self.margin = margin

    def call(self, y_true, y_pred):
        # Since it is automatically converted to two dimensions, specify the shape again.
        y_true = tf.reshape(y_true, [-1])

        # The output is two-dimensional for the VAT.
        y_score = y_pred[:, 1] - y_pred[:, 0]
        positive_score = tf.reshape(y_score[y_true == 1], shape=(-1, 1))
        negative_score = y_score[y_true == 0]

        size = tf.shape(positive_score)[0] * tf.shape(negative_score)[0]
        # if tf.shape(positive_score)[0] * tf.shape(negative_score)[0] == 0:
        #     loss = tf.constant(0.0, dtype=tf.float32)
        # else:
        #     d = negative_score - positive_score
        #     hinge_loss = tf.nn.relu(d + self.margin)
        #     # Corrected so that the minimum value is 0.
        #     loss = tf.math.reduce_mean(tf.math.exp(hinge_loss)) - 1
        loss = tf.cond(
            tf.equal(size, 0),
            true_fn=lambda: tf.constant(0.0, dtype=tf.float32),
            false_fn=lambda: tf.math.reduce_mean(tf.math.exp(tf.nn.relu6(
                negative_score - positive_score + self.margin
            ))) - 1
        )

        return loss

    def get_config(self):
        cfg = super(ExpHingeLoss, self).get_config()
        cfg['margin'] = self.margin
        return cfg


class AucMetric(tf.keras.metrics.Metric):
    def __init__(self, nbins=200, name=None, dtype=None, **kwargs):
        super(AucMetric, self).__init__(name=name, dtype=dtype, **kwargs)

        # https://www.biostat.wisc.edu/~page/rocpr.pdf
        # The implemented in tf.keras.metrics.AUC.
        # That might be more accurate.

        # Calculate the AUC by quantizing the score of each sample in bins.
        # It is not accurate due to quantization.
        # Increase the number of bins to improve the accuracy.
        self.nbins = nbins

        self.positive_bins = self.add_weight(
            'positive_bins', shape=(self.nbins,), dtype=tf.int32,
            initializer=tf.keras.initializers.zeros()
        )
        self.negative_bins = self.add_weight(
            'negative_bins', shape=(self.nbins,), dtype=tf.int32,
            initializer=tf.keras.initializers.zeros()
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Since it is automatically converted to two dimensions, specify the shape again.
        y_true = tf.reshape(y_true, [-1])

        # Since unlabeled data are mixed in for semi-supervised learning,
        # only the labeled portion is taken out.
        # Normalize to [0, 1].
        positive = tf.nn.softmax(y_pred[y_true == 1], axis=1)
        negative = tf.nn.softmax(y_pred[y_true == 0], axis=1)

        tmp_positive = tf.histogram_fixed_width(
            positive[:, 1], value_range=(0, 1), nbins=self.nbins
        )
        tmp_negative = tf.histogram_fixed_width(
            negative[:, 1], value_range=(0, 1), nbins=self.nbins
        )

        self.positive_bins.assign_add(tmp_positive)
        self.negative_bins.assign_add(tmp_negative)

    def result(self):
        # Consider a sample of one positive example.
        # Count the number of negative examples below the positive example.
        # And do it for all the positive examples.
        cum_negative = tf.math.cumsum(self.negative_bins, reverse=False)
        count = self.positive_bins[1:] * cum_negative[:-1]

        n_positive = tf.math.reduce_sum(self.positive_bins)
        n_negative = tf.math.reduce_sum(self.negative_bins)
        n = (tf.cast(n_positive, dtype=tf.float32) *
             tf.cast(n_negative, dtype=tf.float32))
        auc = tf.cast(tf.math.reduce_sum(count), dtype=tf.float32) / n

        return auc

    def reset_states(self):
        if self.built:
            self.positive_bins.assign(np.zeros(self.nbins, dtype=np.int32))
            self.negative_bins.assign(np.zeros(self.nbins, dtype=np.int32))

    def get_config(self):
        d = super(AucMetric, self).get_config()
        d['nbins'] = self.nbins
        return d


class Auc(tf.keras.metrics.Metric):
    def __init__(self, num_thresholds=200, summation_method='interpolation',
                 name=None, dtype=None, thresholds=None, label_weights=None):
        super(Auc, self).__init__()

        self.auc = tf.keras.metrics.AUC(
            num_thresholds=num_thresholds, curve='ROC',
            summation_method=summation_method, name=name,
            dtype=dtype, thresholds=thresholds, multi_label=False,
            label_weights=label_weights
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        flag = tf.math.greater_equal(tf.reshape(y_true, [-1]), 0)
        softmax = tf.math.softmax(y_pred[flag], axis=-1)
        if sample_weight is None:
            w = None
        else:
            w = sample_weight[flag]
        self.auc.update_state(y_true=y_true[flag], y_pred=softmax[:, 1],
                              sample_weight=w)

    def result(self):
        return self.auc.result()

    def reset_states(self):
        self.auc.reset_states()

    def get_config(self):
        return self.auc.get_config()

    @classmethod
    def from_config(cls, config):
        del config['curve'], config['multi_label']
        return Auc(**config)


class PartialExpHingeLoss(tf.keras.losses.Loss):
    def __init__(self, margin, beta, reduction=tf.keras.losses.Reduction.AUTO,
                 name=None):
        super(PartialExpHingeLoss, self).__init__(reduction=reduction,
                                                  name=name)
        self.margin = margin
        self.beta = beta

    def call(self, y_true, y_pred):
        # Since it is automatically converted to two dimensions, specify the shape again.
        y_true = tf.reshape(y_true, [-1])

        # The output is two-dimensional for the VAT.
        y_score = y_pred[:, 1] - y_pred[:, 0]
        positive_score = tf.reshape(y_score[y_true == 1], shape=(-1, 1))
        negative_score = y_score[y_true == 0]

        beta = (tf.cast(tf.shape(negative_score)[0], dtype=tf.float32) *
                self.beta)
        n_beta = tf.cast(tf.math.ceil(beta), dtype=tf.int32)

        values, indices = tf.math.top_k(negative_score, k=n_beta, sorted=False)
        min_element = tf.math.reduce_min(negative_score)

        d = values - positive_score
        hinge_loss = tf.math.exp(tf.nn.relu(d + self.margin))
        loss_all = tf.math.reduce_sum(hinge_loss)

        # The portion of the edge region to be calculated in pAUC.
        d_edge = min_element - positive_score
        hinge_loss_edge = tf.math.exp(tf.nn.relu(d_edge + self.margin))
        loss_outer = (1 - self.beta) * tf.math.reduce_sum(hinge_loss_edge)

        n = tf.cast(tf.shape(positive_score)[0], dtype=tf.float32) * beta
        loss = (loss_all - loss_outer) / n

        return loss

    def get_config(self):
        cfg = super(PartialExpHingeLoss, self).get_config()
        cfg.update({'margin': self.margin, 'beta': self.beta})
        return cfg


class PaucMetric(tf.keras.metrics.Metric):
    def __init__(self, beta, nbins=200, name=None, dtype=None, **kwargs):
        super(PaucMetric, self).__init__(name=name, dtype=dtype, **kwargs)

        self.beta = beta
        self.nbins = nbins

        self.positive_bins = self.add_weight(
            'positive_bins', shape=(self.nbins,), dtype=tf.int32,
            initializer=tf.keras.initializers.zeros()
        )
        self.negative_bins = self.add_weight(
            'negative_bins', shape=(self.nbins,), dtype=tf.int32,
            initializer=tf.keras.initializers.zeros()
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Since it is automatically converted to two dimensions, specify the shape again.
        y_true = tf.reshape(y_true, [-1])

        positive = tf.nn.softmax(y_pred[y_true == 1], axis=1)
        negative = tf.nn.softmax(y_pred[y_true == 0], axis=1)

        tmp_positive = tf.histogram_fixed_width(
            positive[:, 1], value_range=(0, 1), nbins=self.nbins
        )
        tmp_negative = tf.histogram_fixed_width(
            negative[:, 1], value_range=(0, 1), nbins=self.nbins
        )

        self.positive_bins.assign_add(tmp_positive)
        self.negative_bins.assign_add(tmp_negative)

    def result(self):
        
        n_negative = tf.cast(tf.math.reduce_sum(self.negative_bins),
                             dtype=tf.float32)
        beta = n_negative * self.beta

        # Select indices below the threshold beta.
        tmp = tf.cast(tf.math.cumsum(self.negative_bins, reverse=True),
                      dtype=tf.float32)
        flag = tmp > beta

        tmp_negative = tf.where(flag, self.negative_bins, 0)
        cum_negative = tf.math.cumsum(tmp_negative, reverse=False)
        count = self.positive_bins[1:] * cum_negative[:-1]

        n_positive = tf.cast(tf.math.reduce_sum(self.positive_bins),
                             dtype=tf.float32)

        pauc = (tf.cast(tf.math.reduce_sum(count), dtype=tf.float32) /
                (beta * n_positive))

        return pauc

    def reset_states(self):
        if self.built:
            self.positive_bins.assign(np.zeros(self.nbins, dtype=np.int32))
            self.negative_bins.assign(np.zeros(self.nbins, dtype=np.int32))

    def get_config(self):
        d = super(PaucMetric, self).get_config()
        d['beta'] = self.beta
        d['nbins'] = self.nbins
        return d


class PartialAuc(tf.keras.metrics.Metric):
    def __init__(self, beta, num_thresholds=200,
                 summation_method='interpolation', name=None, dtype=None,
                 thresholds=None, label_weights=None):
        super(PartialAuc, self).__init__()

        self.beta = beta

        self.auc = tf.keras.metrics.AUC(
            num_thresholds=num_thresholds, curve='ROC',
            summation_method=summation_method, name=name, dtype=dtype,
            thresholds=thresholds, multi_label=False,
            label_weights=label_weights
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        flag = tf.math.greater_equal(tf.reshape(y_true, [-1]), 0)
        softmax = tf.math.softmax(y_pred[flag], axis=-1)
        if sample_weight is None:
            w = None
        else:
            w = sample_weight[flag]
        self.auc.update_state(y_true=y_true[flag], y_pred=softmax[:, 1],
                              sample_weight=w)

    def result(self):
        # Set `x` and `y` values for the curves based on `curve` config.
        recall = tf.math.divide_no_nan(
            self.auc.true_positives,
            tf.math.add(self.auc.true_positives, self.auc.false_negatives))
        fp_rate = tf.math.divide_no_nan(
            self.false_positives,
            tf.math.add(self.auc.false_positives, self.auc.true_negatives))
        x = fp_rate
        y = recall

        # Find the rectangle heights based on `summation_method`.
        summation_method = self.auc.summation_method
        if summation_method == type(summation_method).INTERPOLATION:
            # Note: the case ('PR', 'interpolation') has been handled above.
            heights = (y[:self.auc.num_thresholds - 1] + y[1:]) / 2.
        elif summation_method == type(summation_method).MINORING:
            heights = tf.minimum(y[:self.auc.num_thresholds - 1], y[1:])
        else:  # summation_method = type(summation_method).MAJORING:
            heights = tf.maximum(y[:self.auc.num_thresholds - 1], y[1:])

        # Find the interval that contains beta.
        flag1 = x[:self.auc.num_thresholds - 1] < self.beta
        flag2 = x[1:] < self.beta
        # The interval where one is true and the other is false contains beta.
        target = tf.math.logical_xor(flag1, flag2)

        # All values in the intervals.
        tmp_heights1 = tf.where(tf.math.logical_and(flag1, flag2),
                                heights, 0.0)
        # Partial to beta.
        delta = self.beta - x[1:]
        tmp_heights2 = tf.where(target, delta * heights, 0.0)

        # Sum up the areas of all the rectangles.
        return tf.math.divide_no_nan(
            tf.reduce_sum(
                tf.multiply(
                    x[:self.auc.num_thresholds - 1] - x[1:],
                    tmp_heights1 + tmp_heights2
                )
            ),
            self.beta,
            name=self.auc.name
        )

    def reset_states(self):
        self.auc.reset_states()

    def get_config(self):
        config = self.auc.get_config()
        config['beta'] = float(self.beta)
        return config

    @classmethod
    def from_config(cls, config):
        del config['curve'], config['multi_label']
        return PartialAuc(**config)


class Crossentropy(tf.keras.losses.Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, name=None):
        super(Crossentropy, self).__init__(reduction=reduction, name=name)

        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=reduction
        )

    def call(self, y_true, y_pred):
        # Since it is automatically converted to two dimensions, specify the shape again.
        # Select only the labeled samples.
        flag = tf.math.not_equal(tf.reshape(y_true, [-1]), -1)
        y_true = y_true[flag]
        y_pred = y_pred[flag]

        return self.loss(y_true=y_true, y_pred=y_pred)


class Accuracy(tf.keras.metrics.Metric):
    def __init__(self, name=None, dtype=None):
        super(Accuracy, self).__init__(name=name, dtype=dtype)

        self.metric = tf.keras.metrics.SparseCategoricalAccuracy(dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Since it is automatically converted to two dimensions, specify the shape again.
        # Select only the labeled samples.
        flag = tf.math.not_equal(tf.reshape(y_true, [-1]), -1)
        y_true = y_true[flag]
        y_pred = y_pred[flag]

        self.metric.update_state(y_true=y_true, y_pred=y_pred)

    def result(self):
        return self.metric.result()

    def reset_states(self):
        self.metric.reset_states()


def make_model(model_type, norm_type=None, input_shape=(29, 29, 3),
               kernel_size=5, padding='same', num_feature_layers=4,
               drop_rate=0.3, base=False,
               use_auc=True, use_vat=True, use_ce=True, **kwargs):
    # unused
    # The required arguments should be listed individually by name.
    _ = kwargs

    inputs = tf.keras.Input(shape=input_shape, dtype=tf.float32)

    #
    # normalization
    #
    if norm_type is None:
        norm_type = model_type

    if norm_type == 'simple':
        h = NormalizeSimple()(inputs)
    elif norm_type == 'middle':
        h = NormalizeMiddle()(inputs)
    elif norm_type == 'complex':
        h = NormalizeComplex()(inputs)
    elif norm_type == 'norm1':
        h = Normalization1()(inputs)
    else:
        raise ValueError(norm_type)

    if model_type == 'complex':
        outputs = make_model_output_complex(
            inputs=h, kernel_size=kernel_size,
            padding=padding, num_feature_layers=num_feature_layers,
            drop_rate=drop_rate
        )
    elif model_type == 'middle':
        outputs = make_model_output_middle(
            inputs=h, kernel_size=kernel_size,
            padding=padding, drop_rate=drop_rate
        )
    elif model_type == 'simple':
        outputs = make_model_output_simple(
            inputs=h, kernel_size=kernel_size, padding=padding,
            drop_rate=drop_rate
        )
    else:
        raise ValueError(model_type)

    if base:
        model = tf.keras.Model(inputs, outputs)
        return model

    # For training.
    model = MyModel(inputs=inputs, outputs=outputs,
                    use_auc=use_auc, use_vat=use_vat, use_ce=use_ce)

    # For prediction.
    tmp = tf.keras.layers.Softmax()(outputs)
    # prediction_outputs = tf.keras.Sequential([
    #     tf.keras.layers.Reshape(target_shape=(-1, 1)),
    #     tf.keras.layers.Cropping1D(cropping=1),
    #     tf.keras.layers.Reshape(target_shape=(-1,))
    # ])(tmp)
    prediction_outputs = tf.keras.layers.Lambda(lambda x: x[:, 1:])(tmp)
    prediction_model = tf.keras.Model(inputs, prediction_outputs)

    return model, prediction_model


def make_model2(model_type, norm_type=None, input_shape=(29, 29, 3),
                kernel_size=5, padding='same', num_feature_layers=4,
                drop_rate=0.3, base=False,
                use_auc=True, use_vat=True, use_ce=True, **kwargs):
    _ = kwargs

    # Mixing Sequential and Functional APIs makes it impossible to load in different versions.

    #
    # normalization
    #
    if norm_type is None:
        norm_type = model_type

    layers = []
    if norm_type == 'simple':
        layers.append(NormalizeSimple(input_shape=input_shape))
    elif norm_type == 'middle':
        layers.append(NormalizeMiddle(input_shape=input_shape))
    elif norm_type == 'complex':
        layers.append(NormalizeComplex(input_shape=input_shape))
    elif norm_type == 'norm1':
        layers.append(Normalization1(input_shape=input_shape))
    else:
        raise ValueError(norm_type)


def _normalize(x):
    # Calculate the mean and variance for spatial direction.
    mean, variance = tf.nn.moments(x, axes=(-2, -3), keepdims=True)
    feature = (x - mean) / (tf.math.sqrt(variance + 1e-6))
    return feature


class NormalizeComplex(tf.keras.layers.Layer):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False,
                 **kwargs):
        super(NormalizeComplex, self).__init__(
            trainable=trainable, name=name, dtype=dtype, dynamic=dynamic,
            **kwargs
        )

    def call(self, inputs, **kwargs):
        normalized1 = _normalize(inputs)
        normalized2 = tf.math.asinh(inputs * 0.5)
        outputs = tf.concat((normalized1, normalized2), axis=-1)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (input_shape[-1] * 2,)


class NormalizeMiddle(tf.keras.layers.Layer):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False,
                 **kwargs):
        super(NormalizeMiddle, self).__init__(
            trainable=trainable, name=name, dtype=dtype, dynamic=dynamic,
            **kwargs
        )

    def call(self, inputs, **kwargs):
        mean, variance = tf.nn.moments(inputs, axes=(-2, -3), keepdims=True)
        std = tf.math.sqrt(variance) + 1e-3
        tmp = tf.clip_by_value(inputs, clip_value_min=mean + 5 * std,
                               clip_value_max=mean + std)

        mean, variance = tf.nn.moments(tmp, axes=(-2, -3), keepdims=True)
        outputs = tf.clip_by_value(
            (inputs - mean) / (tf.math.sqrt(variance) + 1e-3),
            clip_value_min=-5.0, clip_value_max=5.0
        )

        return outputs


class NormalizeSimple(tf.keras.layers.Layer):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False,
                 **kwargs):
        super(NormalizeSimple, self).__init__(
            trainable=trainable, name=name, dtype=dtype, dynamic=dynamic,
            **kwargs
        )

    def call(self, inputs, **kwargs):
        min_value = tf.math.reduce_min(inputs, axis=(-2, -3), keepdims=True)
        max_value = tf.math.reduce_max(inputs, axis=(-2, -3), keepdims=True)
        outputs = (inputs - min_value) / (max_value - min_value)
        return outputs


class Normalization1(tf.keras.layers.Layer):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False,
                 **kwargs):
        super(Normalization1, self).__init__(
            trainable=trainable, name=name, dtype=dtype, dynamic=dynamic,
            **kwargs
        )

    def call(self, inputs, **kwargs):
        tmp = tf.math.asinh(inputs * 0.5)
        min_value = tf.math.reduce_min(tmp, axis=(-2, -3), keepdims=True)
        max_value = tf.math.reduce_max(tmp, axis=(-2, -3), keepdims=True)
        outputs = (tmp - min_value) / (max_value - min_value)
        return outputs



def make_model_output_complex(inputs, kernel_size=5, padding='same',
                              num_feature_layers=4, drop_rate=0.1):
    #
    # making features
    #
    layers = [
        tf.keras.layers.Conv2D(
            filters=32, kernel_size=kernel_size, padding=padding,
            data_format='channels_last', activation='relu'
        )
    ]
    for i in range(num_feature_layers):
        # if i % 2 == 0:
        #     layers.append(
        #         tf.keras.layers.SpatialDropout2D(rate=drop_rate,
        #                                          data_format='channels_last')
        #     )
        layers.append(
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, padding=padding,
                data_format='channels_last', activation='relu'
            )
        )
    layers.extend([
        tf.keras.layers.MaxPooling2D(padding='same'),
        tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same',
                               activation='relu'),
        ResidualBlock(filters=64),
        tf.keras.layers.MaxPooling2D(padding='same'),
        tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same',
                               activation='relu'),
        ResidualBlock(filters=128),
        ResidualBlock(filters=256, strides=2),
        # tf.keras.layers.MaxPooling2D(padding='same'),
        ResidualBlock(filters=256),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(units=2)
    ])

    outputs = tf.keras.Sequential(layers=layers)(inputs)
    return outputs


def make_model_output_middle(inputs, kernel_size=5, padding='same',
                             drop_rate=0.3):
    outputs = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=32, kernel_size=kernel_size,
                               padding=padding, activation=None),
        tf.keras.layers.Activation(tf.keras.activations.swish),
        InceptionBlock(filters=32),
        tf.keras.layers.MaxPooling2D(pool_size=2, padding=padding),
        InceptionBlock(filters=64),
        tf.keras.layers.Activation(tf.keras.activations.swish),
        InceptionBlock(filters=64),
        tf.keras.layers.MaxPooling2D(pool_size=2, padding=padding),
        InceptionBlock(filters=128),
        tf.keras.layers.MaxPooling2D(pool_size=2, padding=padding),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(rate=drop_rate),
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=2)
    ])(inputs)
    return outputs


def make_model_output_simple(inputs, kernel_size=5, padding='valid',
                             drop_rate=0.3):
    # The size of the last output is 2 due to the VAT.
    # Also, there is no activation at the end.
    outputs = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=32, kernel_size=kernel_size,
                               padding=padding, activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=2, padding=padding),
        tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding=padding,
                               activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=2, padding=padding),
        tf.keras.layers.Dropout(rate=drop_rate),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=2)
    ])(inputs)
    return outputs


def make_complex_layers(kernel_size=5, padding='same', num_feature_layers=4):
    layers = [
        tf.keras.layers.Conv2D(
            filters=32, kernel_size=kernel_size, padding=padding,
            data_format='channels_last', activation='relu'
        )
    ]
    for i in range(num_feature_layers):
        layers.append(
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, padding=padding,
                data_format='channels_last', activation='relu'
            )
        )
    layers.extend([
        tf.keras.layers.MaxPooling2D(padding='same'),
        tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same',
                               activation='relu'),
        ResidualBlock(filters=64),
        tf.keras.layers.MaxPooling2D(padding='same'),
        tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same',
                               activation='relu'),
        ResidualBlock(filters=128),
        ResidualBlock(filters=256, strides=2),
        ResidualBlock(filters=256),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(units=2)
    ])

    return layers


def make_middle_layers(kernel_size=5, padding='same', drop_rate=0.3):
    layers = [
        tf.keras.layers.Conv2D(filters=32, kernel_size=kernel_size,
                               padding=padding, activation=None),
        tf.keras.layers.Activation(tf.keras.activations.swish),
        InceptionBlock(filters=32),
        tf.keras.layers.MaxPooling2D(pool_size=2, padding=padding),
        InceptionBlock(filters=64),
        tf.keras.layers.Activation(tf.keras.activations.swish),
        InceptionBlock(filters=64),
        tf.keras.layers.MaxPooling2D(pool_size=2, padding=padding),
        InceptionBlock(filters=128),
        tf.keras.layers.MaxPooling2D(pool_size=2, padding=padding),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(rate=drop_rate),
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=2)
    ]
    return layers


class MyModel(tf.keras.Model):
    def __init__(self, inputs, outputs, use_auc, use_vat, use_ce,
                 epsilon=0.1, xi=1.0, ip=1, name=None):
        super(MyModel, self).__init__()

        self.net = tf.keras.Model(inputs, outputs)

        self.use_auc = use_auc
        self.use_vat = use_vat
        self.use_ce = use_ce

        self.epsilon = epsilon
        self.xi = xi
        self.ip = ip

    def train_step(self, data):
        x, y = data

        y_true = {}
        y_pred = {}
        with tf.GradientTape() as tape:
            p = self.net(x, training=True)
            if self.use_auc:
                y_true['auc'] = y
                y_pred['auc'] = p
            if self.use_ce:
                y_true['ce'] = y
                y_pred['ce'] = p

            if self.use_vat:
                outputs = self.net(x, training=False)
                y_true['vat'] = tf.stop_gradient(
                    tf.math.softmax(outputs, axis=-1)
                )

                vat_perturbation = self.compute_adversarial_noise(
                    inputs=x, outputs=outputs
                )
                noise = tf.stop_gradient(self.epsilon * vat_perturbation)
                p_adv = self.net(x + noise, training=False)
                y_pred['vat'] = p_adv

            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y_true, y_pred,
                                      regularization_losses=self.losses)
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y_true, y_pred)

        # Return a dict mapping metric names to the current value
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data

        y_true = {}
        y_pred = {}

        p = self.net(x, training=False)
        if self.use_auc:
            y_true['auc'] = y
            y_pred['auc'] = p
        if self.use_ce:
            y_true['ce'] = y
            y_pred['ce'] = p

        if self.use_vat:
            y_true['vat'] = tf.stop_gradient(
                tf.math.softmax(p, axis=-1)
            )

            vat_perturbation = self.compute_adversarial_noise(
                inputs=x, outputs=p
            )
            noise = tf.stop_gradient(self.epsilon * vat_perturbation)
            p_adv = self.net(x + noise, training=False)
            y_pred['vat'] = p_adv

        # Compute the loss value
        # (the loss function is configured in `compile()`)
        self.compiled_loss(y_true, y_pred, regularization_losses=self.losses)

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y_true, y_pred)

        # Return a dict mapping metric names to the current value
        return {m.name: m.result() for m in self.metrics}

    def compute_adversarial_noise(self, inputs, outputs):
        def clip(tensor):
            clip_value_min = 1e-6
            return tf.math.maximum(tensor, clip_value_min)

        def normalize(tensor):
            batch_size = tf.shape(tensor)[0]
            t = tf.reshape(tensor, shape=(batch_size, -1))
            n = clip(tf.norm(t, axis=1))
            n = tf.reshape(n, shape=(batch_size, 1, 1, 1))
            return tensor / n

        plain_softmax = tf.nn.softmax(outputs, axis=-1)
        vat_perturbation = normalize(tf.random.normal(tf.shape(inputs)))
        for _ in range(self.ip):
            perturbation = self.xi * vat_perturbation

            with tf.GradientTape() as tape:
                tape.watch(perturbation)

                log_softmax_accommodating_perturbation = tf.nn.log_softmax(
                    self.net(inputs + perturbation), axis=-1
                )
                cross_entropy_accommodating_perturbation = -tf.math.reduce_sum(
                    plain_softmax * log_softmax_accommodating_perturbation,
                    axis=-1
                )
            adversarial_direction = tape.gradient(
                cross_entropy_accommodating_perturbation, [perturbation]
            )[0]
            vat_perturbation = normalize(adversarial_direction)
        return vat_perturbation

    def call(self, inputs, training=None, mask=None):
        outputs = self.net(inputs, training=training, mask=mask)
        return outputs

    def get_config(self):
        # Since ndarray is included in the network structure, it cannot be stored in json.
        # If you want to save the file in json, use "to_json".
        config = self.net.get_config()
        d = {'use_auc': self.use_auc, 'use_vat': self.use_vat,
             'use_ce': self.use_ce, 'epsilon': float(self.epsilon),
             'xi': float(self.xi), 'ip': int(self.ip),
             'network': config, 'lambda_vat': self.lambda_vat}
        return d

    def to_json(self, **kwargs):
        d = self.net.to_json(**kwargs)

        import json

        obj = json.loads(d)
        config = {'use_auc': self.use_auc, 'use_vat': self.use_vat,
                  'use_ce': self.use_ce, 'epsilon': float(self.epsilon),
                  'xi': float(self.xi), 'ip': int(self.ip),
                  'network': obj, 'lambda_vat': self.lambda_vat}

        d = json.dumps(config, **kwargs)
        return d
