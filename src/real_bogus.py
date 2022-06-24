#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import platform
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn.utils
import tensorflow as tf
# import tensorflow_addons as tfa
# from dotenv import dotenv_values
from sacred import Experiment
# from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
from scipy.special import softmax
from sklearn.model_selection import StratifiedKFold, train_test_split

from models.real_bogus.model2 import (Accuracy, Auc, Crossentropy,
                                      ExpHingeLoss, ResidualBlock, make_model)

__date__ = '2021/11/01'


# dotenv_path = Path(__file__).absolute().parents[1] / '.env'
# env = dotenv_values(dotenv_path=dotenv_path, encoding='utf-8')

ex = Experiment('real-bogus-classifier',
                ingredients=[])
# ex.observers.append(MongoObserver(
#     url='mongodb://{user}:{password}@{host}/?authMechanism=SCRAM-SHA-1'.format(
#         user=env['MONGO_USER'], password=env['MONGO_PW'],
#         host=env['MONGO_IP']
#     ),
#     db_name='real-bogus'
# ))
ex.captured_out_filter = apply_backspaces_and_linefeeds


# noinspection PyUnusedLocal
@ex.config
def my_config():
    train_data_dir = None
    # Directory containing test data.
    # Individual file names must be specified for directories other than the predefined ones.
    test_data_dir = None
    # Individual test file names.
    test_real_name, test_bogus_name = None, None
    # Directory for saving the results (predictions, models, and etc.) of the first stage of training.
    filter_result_dir = None
    # Directory for saving the results of the second stage of training.
    output_dir = None
    # Path of the csv file which is the result about the ratio of mislabeling.
    label_error_table_path = '../data/processed/label_error.csv'
    # Directory for the results of the original first stage of training. Label errors are made based on the results.
    base_result_dir = '../models/exp202112/filter_result0'

    batch_size = 256
    model_cfg = {
        # simple or complex
        'model_type': 'complex',
        # normalization type. 'simple' is used for the both model types.
        'norm_type': 'simple',
        'kernel_size': 5,
        # 'valid' for the simple model. 'same' for the complex model.
        'padding': 'same',
        # The number of convolution layers (only for the complex model).
        'num_feature_layers': 4,
        # The rate to drop (make output elements zero) (only for the simple model)
        'drop_rate': 0.3,
        # ID to be specified in "each" case of the experiment. In "mix" and "all" cases, specify -1.
        'detector_id': -1
    }

    train_cfg = {
        'optimizer': 'sgd',
        'lr': 1e-3,
        'optimizer_cfg': {
            'use_clip': False,
            'clip': 'value',
            'value': 1e-1
        },
        'margin': 0.25,
        'auc_nbins': 200,
        # Set a large enough value because of early stopping.
        'epochs': 1000,
        'lambda_auc': 1.0,
        'lambda_vat': 0.1,
        'lambda_ce': 0.1,
        # 0: make mislabeled samples to unlabeled.
        # 1: (unused) make mislabeled samples whose probabilities are larger than the threshold to be unlabeled.
        # 2: All samples are used as they are with labels.
        'selection_mode': 0,
        # The threshold that is used when selection_mode is 1.
        'selection_threshold': 0.7,
        # Whether to re-label mislabeled samples or not. "none" for not to relabel. "pseudo" to relabel.
        'label_mode': 'none',
        # Increase label errors to include original label errors.
        # A multiplication factor: how many times the label error is increased as compared with the original.
        'label_noise': -1.0,
        'label_noise_seed': 42,
        # early stopping condition. see https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping
        'patience': 10,
        # Set True in "mix" case.
        'small_mix_dataset': False,
        # Specify a fold number if the specific fold is to be learned in the first stage of training.
        'target_fold': None
    }
    n_splits = 5
    # The option to save the model in the first stage and "convert_to_hdf5".
    save_weights_only = False
    # The file format to save. tf or h5.
    save_format = 'tf'

    # Seed value for the first stage of training.
    base_classifier_seed = 0x5eed

    # 0: show nothing
    # 1: show the progress updated at each batch
    # 2: show the progress updated at each epoch
    if platform.system() == 'Windows':
        verbose = 1
    else:
        verbose = 2


def set_seed(seed):
    tf.random.set_seed(seed)

    # optional
    # for numpy.random
    np.random.seed(seed)
    # for built-in random
    random.seed(seed)
    # for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


@ex.capture
def load_train_data_detector(train_data_dir, detector_id, batch_size, seed):
    data_dir = Path(train_data_dir)
    record_path = data_dir / 'data{}.tfrecord'.format(detector_id)

    dataset = tf.data.TFRecordDataset([str(record_path)],
                                      compression_type='GZIP')
    dataset = dataset.map(
        lambda x: map_record(parse(x)),
        deterministic=True,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    ).batch(1024)

    (train_x, train_y), (val_x, val_y), _ = split_dataset(
        dataset=dataset, seed1=seed, seed2=seed + 1
    )

    train_ds = tf.data.Dataset.from_tensor_slices(
        (train_x, train_y)
    ).shuffle(batch_size * 8).batch(batch_size)
    val_ds = tf.data.Dataset.from_tensor_slices(
        (val_x, val_y)
    ).batch(batch_size)

    return train_ds, val_ds


@ex.capture
def load_train_data_mix_detector(train_data_dir, batch_size):
    """
    Prepare mixed training data with the same size as the each detector data.

    To test whether mixed data or increasing volume is more effective.
    Args:
        train_data_dir:
        batch_size:

    Returns:

    """
    data_dir = Path(train_data_dir)
    data_info = pd.read_csv(data_dir / 'data_info.csv', header=0, index_col=0)
    mean_size = data_info['size'].mean()
    total_size = data_info['size'].sum()

    train_image_list, train_label_list = [], []
    val_image_list, val_label_list = [], []
    for _, row in data_info.iterrows():
        detector_id = row['detector_id']
        record_path = data_dir / 'data{}.tfrecord'.format(detector_id)

        # Nunmbe of samples to pick up from this data.
        n = int(np.round(row['size'] / total_size * mean_size))

        dataset = tf.data.TFRecordDataset(
            [str(record_path)], compression_type='GZIP'
        ).shuffle(row['size'], seed=row['detector_id']).take(n).map(
            lambda x: map_record(parse(x)),
            deterministic=True,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        ).batch(1024)

        (train_x, train_y), (val_x, val_y), _ = split_dataset(
            dataset=dataset, seed1=detector_id, seed2=detector_id + 1
        )

        train_x, train_y = sklearn.utils.shuffle(
            train_x, train_y, random_state=detector_id + 2
        )
        val_x, val_y = sklearn.utils.shuffle(
            val_x, val_y, random_state=detector_id + 3
        )

        train_image_list.append(train_x)
        train_label_list.append(train_y)
        val_image_list.append(val_x)
        val_label_list.append(val_y)

    train_image_list = np.concatenate(train_image_list, axis=0)
    train_label_list = np.concatenate(train_label_list, axis=0)
    val_image_list = np.concatenate(val_image_list, axis=0)
    val_label_list = np.concatenate(val_label_list, axis=0)

    train_ds = tf.data.Dataset.from_tensor_slices(
        (train_image_list, train_label_list)
    ).shuffle(batch_size * 16).batch(batch_size=batch_size).prefetch(
        tf.data.experimental.AUTOTUNE
    )
    val_ds = tf.data.Dataset.from_tensor_slices(
        (val_image_list, val_label_list)
    ).batch(batch_size=batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    return train_ds, val_ds


def split_dataset(dataset, seed1, seed2):
    """
    Split tf.data.Dataset with ratio 5 : 1 : 1 and return them as numpy format.
    Args:
        dataset:
        seed1:
        seed2:

    Returns:

    """
    images, labels = [], []
    for image, label in dataset:
        images.append(image)
        labels.append(label)
    images = tf.concat(images, axis=0).numpy()
    labels = tf.concat(labels, axis=0).numpy()

    # split data; train : val : test = 5 : 1 : 1 .
    # test data is not used.
    train_x, tmp_x, train_y, tmp_y = train_test_split(
        images, labels, test_size=2.0 / 7.0, stratify=labels,
        random_state=seed1, shuffle=True
    )
    val_x, test_x, val_y, test_y = train_test_split(
        tmp_x, tmp_y, test_size=0.5, stratify=tmp_y,
        random_state=seed2, shuffle=True
    )

    return (train_x, train_y), (val_x, val_y), (test_x, test_y)


@ex.capture
def load_train_data(train_data_dir, noise_indices, train_cfg, batch_size,
                    seed):
    data_dir = Path(train_data_dir)

    data_info = pd.read_csv(data_dir / 'data_info.csv', header=0, index_col=0)

    total_size = data_info['size'].sum()
    index_list = np.arange(total_size, dtype=np.int32)

    # records = tf.io.gfile.glob('{}/*.tfrecord'.format(data_dir))
    dataset = tf.data.Dataset.list_files('{}/*.tfrecord'.format(data_dir))
    dataset = dataset.interleave(
        lambda x: tf.data.TFRecordDataset(x, compression_type='GZIP'),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    dataset = dataset.map(
        parse,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    train_index, val_index = train_test_split(
        index_list, shuffle=True, random_state=seed
    )

    train_flag = np.zeros(total_size, dtype=bool)
    val_flag = np.zeros_like(train_flag)

    train_flag[train_index] = True
    val_flag[val_index] = True

    if noise_indices is not None:
        noise_flag = np.zeros(total_size, dtype=bool)
        noise_flag[noise_indices] = True

        noise_flag = tf.constant(noise_flag)
        label_mode = train_cfg['label_mode']
        if label_mode == 'none':
            train_ds = make_dataset_semi_supervised(
                dataset=dataset, flag=train_flag, batch_size=batch_size,
                shuffle=True, cache=False, noise_flag=noise_flag
            )
            val_ds = make_dataset_semi_supervised(
                dataset=dataset, flag=val_flag, batch_size=batch_size,
                shuffle=False, cache=True, noise_flag=noise_flag
            )
        elif label_mode == 'pseudo':
            train_ds = make_dataset_pseudo_label(
                dataset=dataset, flag=train_flag, batch_size=batch_size,
                shuffle=True, cache=False, noise_flag=noise_flag
            )
            val_ds = make_dataset_pseudo_label(
                dataset=dataset, flag=val_flag, batch_size=batch_size,
                shuffle=False, cache=True, noise_flag=noise_flag
            )
        else:
            raise ValueError(label_mode)
    elif train_cfg['label_noise'] > 0:
        # For supervised learning with all teacher labels and increased label errors.
        # Since it is difficult to find out which samples have inverted labels, 
        # we use the label information in validation_result.csv instead of the labels in tfrecords.
        df = load_validation_result()
        df.sort_index(inplace=True)

        label_list = tf.constant(df['label'].values)

        train_ds = make_dataset_label(
            dataset=dataset, flag=train_flag, label_list=label_list,
            batch_size=batch_size, shuffle=True, cache=False
        )
        val_ds = make_dataset_label(
            dataset=dataset, flag=val_flag, label_list=label_list,
            batch_size=batch_size, shuffle=False, cache=True
        )
    else:
        train_ds = make_dataset(
            dataset=dataset, flag=train_flag, batch_size=batch_size,
            shuffle=True, cache=False
        )
        val_ds = make_dataset(
            dataset=dataset, flag=val_flag, batch_size=batch_size,
            shuffle=False, cache=True
        )

    return train_ds, val_ds


@tf.function
def parse(example_proto):
    feature_description = {
        'image': tf.io.FixedLenFeature(shape=[29 * 29 * 3], dtype=tf.float32),
        'label': tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
        'detector_id': tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
        'sample_index': tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
        'unique_index': tf.io.FixedLenFeature(shape=[], dtype=tf.int64)
    }
    return tf.io.parse_single_example(example_proto, feature_description)


@ex.capture
def make_label_error(base_result_dir, n_splits, label_error_table_path,
                     train_cfg):
    df = pd.read_csv(label_error_table_path, index_col=0, header=0)

    base_result_dir = Path(base_result_dir)
    df_list = []
    for i in range(n_splits):
        tmp = pd.read_csv(base_result_dir / str(i) / 'validation_result.csv',
                          index_col=0, header=0)
        df_list.append(tmp)
    table = pd.concat(df_list, axis=0)
    table['score'] = table['prediction1'] - table['prediction0']

    range1, range2, probability = make_target_ratio(df=df)

    noise_flag = pd.Series(np.zeros(len(table), dtype=bool))
    ratio = train_cfg['label_noise'] - 1.0
    generator = np.random.default_rng(seed=train_cfg['label_noise_seed'])
    for r1, r2, p in zip(range1, range2, probability):
        tmp = table[np.logical_and(table['score'] >= r1, table['score'] < r2)]
        n_positive = np.count_nonzero(tmp['label'] == 1)
        n_negative = np.count_nonzero(tmp['label'] == 0)

        if r1 >= 0:
            n_error = n_negative * p * ratio
            data = tmp[tmp['label'] == 1]
        else:
            n_error = n_positive * p * ratio
            data = tmp[tmp['label'] == 0]
        selected = generator.choice(data.index,
                                    size=min(int(n_error), len(data)),
                                    replace=False)
        noise_flag[selected] = True
    # Sort by index to be sure.
    noise_flag.sort_index(inplace=True, ascending=True)

    return noise_flag.values


def make_target_ratio(df):
    r1 = df['r1']
    r2 = df['r2']
    p = df['p']

    tmp_r1a, tmp_r2a, tmp_pa = [], [], []
    for x in range(-15, -4):
        tmp_r1a.append(x)
        tmp_r2a.append(x + 1)
        tmp_pa.append(1)
    tmp_r1b, tmp_r2b, tmp_pb = [], [], []
    for x in range(4, 15):
        tmp_r1b.append(x)
        tmp_r2b.append(x + 1)
        tmp_pb.append(1)
    r1 = np.hstack((tmp_r1a, r1, tmp_r1b))
    r2 = np.hstack((tmp_r2a, r2, tmp_r2b))
    p = np.hstack((tmp_pa, p, tmp_pb))

    return r1, r2, p


@ex.capture
def load_train_data_kfold(train_data_dir, batch_size, n_splits, seed,
                          train_cfg):
    data_dir = Path(train_data_dir)

    data_info = pd.read_csv(data_dir / 'data_info.csv', header=0, index_col=0)

    total_size = data_info['size'].sum()
    
    index_list = []
    for _, row in data_info.iterrows():
        index_list.extend([row['detector_id']] * row['size'])

    records = tf.io.gfile.glob('{}/*.tfrecord'.format(data_dir))

    if train_cfg['label_noise'] > 0:
        # This flag is used to make noisy training data, 
        # and the samples are purposely chosen to invert the labels.
        noise_flag = make_label_error()
    else:
        noise_flag = None
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_index, val_index in skf.split(index_list, index_list):
        train_flag = np.zeros(total_size, dtype=bool)
        val_flag = np.zeros_like(train_flag)

        train_flag[train_index] = True
        val_flag[val_index] = True

        dataset = tf.data.TFRecordDataset(
            records, compression_type='GZIP',
            buffer_size=None,
            num_parallel_reads=tf.data.experimental.AUTOTUNE
        )
        dataset = dataset.map(
            parse,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        if noise_flag is None:
            train_ds = make_dataset(
                dataset=dataset, flag=train_flag, batch_size=batch_size,
                shuffle=True, cache=False
            )
            val_ds = make_dataset(
                dataset=dataset, flag=val_flag, batch_size=batch_size,
                shuffle=False, cache=True
            )
            evaluation_ds = make_dataset2(dataset=dataset, flag=val_flag,
                                          batch_size=batch_size)
        else:
            train_ds = make_dataset_pseudo_label(
                dataset=dataset, flag=train_flag, noise_flag=noise_flag,
                batch_size=batch_size, shuffle=True, cache=False
            )
            val_ds = make_dataset_pseudo_label(
                dataset=dataset, flag=val_flag, noise_flag=noise_flag,
                batch_size=batch_size, shuffle=False, cache=True
            )

            evaluation_ds = make_dataset2_noise(
                dataset=dataset, flag=val_flag, noise_flag=noise_flag,
                batch_size=batch_size
            )

        d = {'train': train_ds, 'val': val_ds, 'evaluation': evaluation_ds}
        yield d


def make_dataset(dataset, flag, batch_size, shuffle, cache):
    return _make_dataset(
        dataset=dataset, flag=flag, batch_size=batch_size,
        shuffle=shuffle, cache=cache, map_func=map_record
    )


def _make_dataset(dataset, flag, batch_size, shuffle, cache, map_func):
    flag = tf.constant(flag)
    ds = dataset.filter(lambda x: flag[x['unique_index']])

    if cache:
        ds = dataset.cache()
    ds = ds.map(
        map_func=map_func,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        deterministic=False
    )
    if shuffle:
        ds = ds.shuffle(batch_size * 4)
    ds = ds.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    return ds


def map_record(data):
    image = tf.reshape(data['image'], shape=[29, 29, 3])
    label = data['label']

    return image, label


def make_dataset_pseudo_label(dataset, flag, noise_flag, batch_size,
                              shuffle, cache):
    return _make_dataset(
        dataset=dataset, flag=flag, batch_size=batch_size, shuffle=shuffle,
        cache=cache, map_func=lambda data: map_record_pseudo_label(
            data=data, noise_flag=noise_flag
        )
    )


@tf.function
def map_record_pseudo_label(data, noise_flag):
    image = tf.reshape(data['image'], shape=[29, 29, 3])
    label = tf.cond(noise_flag[data['unique_index']],
                    true_fn=lambda: 1 - data['label'],
                    false_fn=lambda: data['label'])
    return image, label


def make_dataset_semi_supervised(dataset, flag, noise_flag, batch_size,
                                 shuffle, cache):
    return _make_dataset(
        dataset=dataset, flag=flag, batch_size=batch_size,
        shuffle=shuffle, cache=cache,
        map_func=lambda data: map_record_semi_supervised(
            data=data, noise_flag=noise_flag
        )
    )


@tf.function
def map_record_semi_supervised(data, noise_flag):
    image = tf.reshape(data['image'], shape=[29, 29, 3])
    label = tf.cond(noise_flag[data['unique_index']],
                    true_fn=lambda: tf.constant(-1, dtype=tf.int64),
                    false_fn=lambda: data['label'])
    return image, label


def make_dataset2(dataset, flag, batch_size):
    flag = tf.constant(flag)
    ds = dataset.filter(
        lambda x: flag[x['unique_index']]
    ).map(
        map_func=map_record2,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        deterministic=False
    ).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    return ds


def map_record2(data):
    image = tf.reshape(data['image'], shape=[29, 29, 3])
    data['image'] = image
    return data


def make_dataset2_noise(dataset, flag, noise_flag, batch_size):
    flag = tf.constant(flag)
    noise_flag = tf.constant(noise_flag)
    ds = dataset.filter(
        lambda x: flag[x['unique_index']]
    ).map(
        map_func=lambda x: map_record2_noise(x, noise_flag=noise_flag),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
        deterministic=False
    ).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    return ds


def map_record2_noise(data, noise_flag):
    image = tf.reshape(data['image'], shape=[29, 29, 3])
    data['image'] = image
    data['label'] = tf.cond(noise_flag[data['unique_index']],
                            true_fn=lambda: 1 - data['label'],
                            false_fn=lambda: data['label'])
    return data


def make_dataset_label(dataset, flag, label_list, batch_size, shuffle, cache):
    """
    Use label_list instead of the labels in tfrecords.
    When supervised learning is performed in a situation where
    the label error is artificially increased and the label is trusted.
    Args:
        dataset:
        flag:
        label_list:
        batch_size:
        shuffle:
        cache:

    Returns:

    """
    return _make_dataset(
        dataset=dataset, flag=flag, batch_size=batch_size,
        shuffle=shuffle, cache=cache,
        map_func=lambda data: map_record_label(
            data=data, label_list=label_list
        )
    )


def map_record_label(data, label_list):
    """
    The labels in the label_list, not the labels in the data, are returned.
    Args:
        data:
        label_list:

    Returns:

    """
    image = tf.reshape(data['image'], shape=[29, 29, 3])
    label = label_list[data['unique_index']]
    return image, label


@ex.capture
def _load_artificial_data(data_dir):
    data_dir = Path(data_dir)

    # Sort by name for reproducibility and fix the order uniquely.
    file_list = list(str(data_dir.glob('images*.npy')))
    file_list.sort()

    r = re.compile(r'images(\d+)\.npy')
    d = {'rand_artificial_real': 1, 'galx_artificial_real': 1, 'artifact': 0}

    image_list = []
    label_list = []
    for image_file in file_list:
        images = np.load(image_file)
        image_list.append(images.astype(np.float32))

        m = r.search(image_file)
        detector_id = m.group(1)
        df = pd.read_csv(data_dir / 'params{}.csv'.format(detector_id),
                         header=0, index_col=None)
        labels = df['object_type'].map(d)
        labels = labels.values.reshape([-1, 1]).astype(np.int32)
        label_list.append(labels)
    images = np.concatenate(image_list, axis=0)
    labels = np.concatenate(label_list, axis=0)

    return images, labels


@ex.capture
def load_test_data(test_data_dir, batch_size, test_real_name, test_bogus_name):
    data_dir = Path(test_data_dir)

    if data_dir.stem == 'testset_201221':
        images, labels = _load_testset_201221(data_dir=data_dir)
    elif data_dir.stem == 'testset_210512':
        images, labels = _load_testset_210512(data_dir=data_dir)
    else:
        if test_real_name is None or test_bogus_name is None:
            raise ValueError(test_data_dir)
        images, labels = _load_test_npy(data_dir=data_dir,
                                        real_name=test_real_name,
                                        bogus_name=test_bogus_name)

    test_ds = tf.data.Dataset.from_tensor_slices((images, labels))
    test_ds = test_ds.batch(batch_size)

    return test_ds


def _load_test_npy(data_dir, real_name, bogus_name):
    # noinspection PyTypeChecker
    real_images = np.load(data_dir / real_name)
    # noinspection PyTypeChecker
    bogus_images = np.load(data_dir / bogus_name)

    images = np.concatenate((real_images, bogus_images),
                            axis=0).astype(np.float32)
    labels = np.array([1] * len(real_images) + [0] * len(bogus_images),
                      dtype=np.int32)
    return images, labels


def _load_testset_201221(data_dir):
    return _load_test_npy(data_dir=data_dir,
                          real_name='real_transient_201217.npy',
                          bogus_name='bogus_transient_201221.npy')


def _load_testset_210512(data_dir):
    return _load_test_npy(data_dir=data_dir,
                          real_name='real_transient_210512_test.npy',
                          bogus_name='bogus_transient_210512_test.npy')


@ex.capture
def make_base_classifier(filter_result_dir, model_cfg, train_cfg,
                         base_classifier_seed, save_weights_only, verbose):
    """
    Train a classifier (the first stage of training).
    Args:
        filter_result_dir:
        model_cfg:
        train_cfg:
        base_classifier_seed:
        save_weights_only:
        verbose:
    """
    filter_result_dir = Path(filter_result_dir)
    if not filter_result_dir.exists():
        filter_result_dir.mkdir(parents=True)

    # dataset_list = load_train_data_kfold()
    test_ds = load_test_data()

    for i, dataset in enumerate(load_train_data_kfold(
            seed=base_classifier_seed)):
        if (train_cfg['target_fold'] is not None and
                train_cfg['target_fold'] != i):
            continue
        set_seed(base_classifier_seed + i)

        # Since the model continues to remain internal, it is deleted.
        tf.keras.backend.clear_session()

        print('fold: {}'.format(i))

        result_dir = filter_result_dir / str(i)
        if not result_dir.exists():
            result_dir.mkdir(parents=True)

        if (result_dir / 'validation_result.csv').exists():
            continue

        checkpoint = result_dir / 'model.ckpt'
        if (checkpoint / 'saved_model.pb').exists():
            model = tf.keras.models.load_model(
                str(checkpoint),
                custom_objects={'ResidualBlock': ResidualBlock}
            )
        else:
            model = make_model(base=True, **model_cfg)
            optimizer = tf.keras.optimizers.SGD(learning_rate=train_cfg['lr'])
            model.compile(
                optimizer=optimizer,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True
                ),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
            )

            if (result_dir / 'checkpoint').exists():
                model.load_weights(str(checkpoint))

        train_ds = dataset['train']
        val_ds = dataset['val']

        callbacks = [
            tf.keras.callbacks.CSVLogger(filename=str(result_dir / 'log.txt')),
            tf.keras.callbacks.TensorBoard(
                log_dir=str(result_dir), write_graph=False, profile_batch=0
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(result_dir / 'model.ckpt'),
                save_best_only=True, save_weights_only=save_weights_only
            ),
            tf.keras.callbacks.EarlyStopping(restore_best_weights=True,
                                             patience=train_cfg['patience'])
        ]
        model.fit(
            train_ds, epochs=train_cfg['epochs'], callbacks=callbacks,
            validation_data=val_ds, verbose=verbose
        )

        # Evaluated with test data for reference.
        predictions = []
        labels = []
        for x, y in test_ds:
            p = model(x, training=False)
            predictions.append(p)
            labels.append(y)
        predictions = tf.concat(predictions, axis=0).numpy()
        labels = tf.concat(labels, axis=0).numpy()
        df = pd.DataFrame(predictions, columns=['prediction0', 'prediction1'])
        df['label'] = labels
        # noinspection PyTypeChecker
        df.to_csv(result_dir / 'test_result.csv')

        # Evaluation of validation data.
        predictions = []
        labels = []
        detector_id_list = []
        sample_index_list = []
        unique_index_list = []
        for data in dataset['evaluation']:
            p = model(data['image'], training=False)
            predictions.append(p)
            labels.append(data['label'])
            detector_id_list.append(data['detector_id'])
            sample_index_list.append(data['sample_index'])
            unique_index_list.append(data['unique_index'])
        predictions = tf.concat(predictions, axis=0).numpy()
        labels = tf.concat(labels, axis=0).numpy()
        detector_id_list = tf.concat(detector_id_list, axis=0).numpy()
        sample_index_list = tf.concat(sample_index_list, axis=0).numpy()
        unique_index_list = tf.concat(unique_index_list, axis=0).numpy()
        df = pd.DataFrame(predictions, columns=['prediction0', 'prediction1'],
                          index=unique_index_list)
        df['label'] = labels
        df['detector_id'] = detector_id_list
        df['sample_index'] = sample_index_list
        df.sort_index(inplace=True)
        # noinspection PyTypeChecker
        df.to_csv(result_dir / 'validation_result.csv')


@ex.capture
def load_validation_result(filter_result_dir, n_splits):
    filter_result_dir = Path(filter_result_dir)

    df_list = []
    for i in range(n_splits):
        tmp = pd.read_csv(filter_result_dir / str(i) / 'validation_result.csv',
                          index_col=0, header=0)
        df_list.append(tmp)
    df = pd.concat(df_list, axis=0)

    return df


@ex.capture
def select_incorrect_index(train_cfg):
    df = load_validation_result()

    selection_mode = train_cfg['selection_mode']
    if selection_mode == 0:
        prediction = np.argmax(df[['prediction0', 'prediction1']].values,
                               axis=1)
        incorrect = df.index[prediction != df['label']]
    elif selection_mode == 1:
        # Samples, for which the degree of error is greater than the threshold, are selected as unlabeled.
        incorrect = select_sample(threshold=train_cfg['selection_threshold'],
                                  df=df)
    elif selection_mode == 2:
        incorrect = None
    else:
        # An unexpected mode was specified.
        raise RuntimeError(selection_mode)

    return incorrect


def select_sample(threshold, df):
    """
    Select samples to be unlabeled based on the classifier's evaluation.

    Args:
        threshold: float
            The threshold for the evaluation of the sample to be selected (usually a value between 0.5 and 1.0).
            The closer to 1.0 the sample is selected, the greater the degree of error is.
            0.5 would select all the misclassified samples.
        df:

    Returns:

    """
    tmp = df[['prediction0', 'prediction1']].values
    probability = softmax(tmp, axis=-1)
    p = probability[:, 1]

    # Select a sample that is misclassified and the degree is above the threshold.
    false_positive = np.logical_and(df['label'] == 0, p > threshold)
    false_negative = np.logical_and(df['label'] == 1, p < 1 - threshold)
    flag = np.logical_or(false_positive, false_negative)

    indices = df.index[flag]
    return indices


@ex.capture
def train_classifier(output_dir, noise_indices, model_cfg, train_cfg, seed,
                     save_format, verbose):
    """
    Training the classifier (the second state of training).
    Args:
        output_dir:
        noise_indices:
        model_cfg:
        train_cfg:
        seed:
        save_format:
        verbose:
    """
    set_seed(seed)

    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    tf.keras.backend.clear_session()

    model, prediction_model = make_model(
        use_auc=train_cfg['lambda_auc'] != 0,
        use_vat=train_cfg['lambda_vat'] != 0,
        use_ce=train_cfg['lambda_ce'] != 0,
        **model_cfg
    )   # type: tf.keras.Model, tf.keras.Model
    if train_cfg['optimizer'] == 'sgd':
        kwargs = dict(learning_rate=train_cfg['lr'])
        if train_cfg['optimizer_cfg']['use_clip']:
            if train_cfg['optimizer_cfg']['clip'] == 'value':
                kwargs['clipvalue'] = train_cfg['optimizer_cfg']['value']
            else:
                kwargs['clipvnorm'] = train_cfg['optimizer_cfg']['value']
        optimizer = tf.keras.optimizers.SGD(**kwargs)
    elif train_cfg['optimizer'] == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=train_cfg['lr'])
    elif train_cfg['optimizer'] == 'adadelta':
        optimizer = tf.keras.optimizers.Adadelta(learning_rate=train_cfg['lr'])
    # elif train_cfg['optimizer'] == 'radam':
    #     optimizer = tfa.optimizers.RectifiedAdam(
    #         learning_rate=train_cfg['lr'],
    #         total_steps=100000,
    #         warmup_proportion=0.1,
    #         min_lr=train_cfg['lr'] * 0.01
    #     )
    # elif train_cfg['optimizer'] == 'novograd':
    #     optimizer = tfa.optimizers.NovoGrad(learning_rate=train_cfg['lr'])
    else:
        raise ValueError(train_cfg['optimizer'])
    # Set loss functions and metrics
    loss = {}
    metrics = {}
    loss_weights = {}
    if train_cfg['lambda_auc'] != 0:
        loss['auc'] = ExpHingeLoss(margin=train_cfg['margin'])
        metrics['auc'] = Auc()
        loss_weights['auc'] = train_cfg['lambda_auc']
    if train_cfg['lambda_vat'] != 0:
        loss['vat'] = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        metrics['vat'] = tf.keras.metrics.CategoricalCrossentropy(
            from_logits=True, name='lds'
        )
        loss_weights['auc'] = train_cfg['lambda_vat']
    if train_cfg['lambda_ce'] != 0:
        loss['ce'] = Crossentropy()
        metrics['ce'] = Accuracy()
        loss_weights['ce'] = train_cfg['lambda_ce']
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
        loss_weights=loss_weights,
        run_eagerly=False
    )

    checkpoint = output_dir / 'model.ckpt.index'
    if checkpoint.exists():
        model.load_weights(
            filepath=str(output_dir / 'model.ckpt')
        ).expect_partial()

    if model_cfg['detector_id'] >= 0:
        train_ds, val_ds = load_train_data_detector(
            detector_id=model_cfg['detector_id']
        )
    elif train_cfg['small_mix_dataset']:
        train_ds, val_ds = load_train_data_mix_detector()
    else:
        train_ds, val_ds = load_train_data(noise_indices=noise_indices)

    callbacks = [
        tf.keras.callbacks.CSVLogger(filename=str(output_dir / 'log.txt')),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(output_dir), write_graph=False, profile_batch=0
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / 'model.ckpt'),
            save_best_only=True, save_weights_only=True
        ),
        tf.keras.callbacks.EarlyStopping(restore_best_weights=True,
                                         patience=train_cfg['patience'])
    ]
    model.fit(
        train_ds, epochs=train_cfg['epochs'], callbacks=callbacks,
        validation_data=val_ds, verbose=verbose
    )

    if save_format == 'tf':
        name = str(output_dir)
    else:
        name = str(output_dir / 'prediction_model.h5')
    prediction_model.save(name, save_format=save_format)


    # Evaluate performance with test data.
    test_ds = load_test_data()
    predictions = []
    labels = []
    for x, y in test_ds:
        p = model(x, training=False)
        predictions.append(p)
        labels.append(y)
    predictions = tf.concat(predictions, axis=0).numpy()
    labels = tf.concat(labels, axis=0).numpy()
    df = pd.DataFrame(predictions, columns=['prediction0', 'prediction1'])
    df['label'] = labels
    # noinspection PyTypeChecker
    df.to_csv(output_dir / 'test_result.csv')


@ex.command
def predict(output_dir, save_format, test_data_dir):
    # Model for prediction
    if save_format == 'tf':
        model = tf.keras.models.load_model(output_dir)
    else:
        model = tf.keras.models.load_model(
            str(Path(output_dir) / 'prediction_model.h5'),
            custom_objects={'ResidualBlock': ResidualBlock}
        )

    test_ds = load_test_data()
    predictions = []
    labels = []
    for x, y in test_ds:
        p = model(x, training=False)
        predictions.append(p)
        labels.append(y)
    predictions = tf.squeeze(tf.concat(predictions, axis=0)).numpy()
    labels = tf.concat(labels, axis=0).numpy()

    name = Path(test_data_dir).stem
    df = pd.DataFrame({'probability': predictions, 'label': labels})
    df.to_csv(Path(output_dir) / 'prediction_{}.csv'.format(name))


@ex.command
def convert_to_hdf5(output_dir, model_cfg, save_weights_only):
    # Since it is not used for learning, the objective function is set at a value for now.
    model, prediction_model = make_model(
        use_auc=False, use_vat=False, use_ce=True, **model_cfg
    )  # type: tf.keras.Model, tf.keras.Model

    print('loading saved model')
    model2 = tf.keras.models.load_model(output_dir)  # type: tf.keras.Model

    for v1, v2 in zip(model.trainable_variables, model2.trainable_variables):
        v1.assign(v2)

    print('saving hdf5')
    output_path = Path(output_dir) / 'prediction_model.h5'
    if save_weights_only:
        prediction_model.save_weights(filepath=str(output_path))
    else:
        prediction_model.save(str(output_path))


@ex.main
def run(train_cfg):
    # 1. Training a classifier to evaluate and select the training data.
    # 2. Selecting samples under a condition and unlabeling.
    # 3. Semi-supervised learning.

    # Evaluate the data left for validation by learning with the training data.
    # It is time consuming to perform cross validation.
    make_base_classifier()
    if train_cfg['target_fold'] is not None:
        return

    print('extracting incorrect indices')
    # Find the index of misclassified data based on the results of 
    # evaluation of validation data in cross-validation.
    incorrect_indices = select_incorrect_index()

    print('training classifier')
    train_classifier(noise_indices=incorrect_indices)


if __name__ == '__main__':
    ex.run_commandline()
