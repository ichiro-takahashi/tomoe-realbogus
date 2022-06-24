#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
To read sequentially the data from the hard disk, the data format is convert to TFRecord.
"""

import re
from pathlib import Path

import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.utils import shuffle

__date__ = '2021/04/06'


def load_data(npy_path, csv_path):
    images = np.load(npy_path)
    df = pd.read_csv(csv_path, index_col=None, header=0)

    d = {'artifact': 0, 'galx_artificial_real': 1, 'rand_artificial_real': 1}
    labels = df['object_type'].map(d).values

    return images, labels


def _bytes_feature(value):
    """Returns byte_list type from string / byte type."""
    if isinstance(value, type(tf.constant(0))):
        # BytesList won't unpack a string from an EagerTensor.
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
    """Returns float_list type from float / double type."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Return Int64_list type from bool / enum / int / uint type."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def make_example(image, label, detector_id, sample_index, unique_index):
    """Convert data formats."""
    feature = {
        'image': _float_feature(image.reshape(-1)),
        'label': _int64_feature([label]),
        'detector_id': _int64_feature([detector_id]),
        'sample_index': _int64_feature([sample_index]),
        'unique_index': _int64_feature([unique_index])
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def main():
    data_dir = Path('../../data/raw/real_bogus1')
    npy_list = list(data_dir.glob('images*.npy'))
    npy_list.sort()

    output_dir = Path('../../data/processed/real_bogus1')
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    r = re.compile(r'images(\d+)')
    unique_id = 0
    # Size and start of unique index for each detector.
    data_info = {'detector_id': [], 'size': [], 'start_index': []}
    for npy_path in npy_list:
        m = r.search(npy_path.stem)
        detector_id = int(m.group(1))

        csv_path = data_dir / 'params{}.csv'.format(detector_id)

        images, labels = load_data(npy_path=npy_path, csv_path=csv_path)
        n = len(images)
        indices = np.arange(n)
        # Unique index across the entire data set.
        unique_indices = indices + unique_id

        data_info['detector_id'].append(detector_id)
        data_info['size'].append(n)
        data_info['start_index'].append(unique_id)

        unique_id += n

        images, labels, indices, unique_indices = shuffle(
            images, labels, indices, unique_indices
        )

        # Write TFRecord.
        record_path = str(output_dir / 'data{}.tfrecord'.format(detector_id))
        with tf.io.TFRecordWriter(
                record_path,
                tf.io.TFRecordOptions(compression_type='GZIP')) as writer:
            for image, label, index, unique_index in zip(
                    tqdm(images, desc=str(detector_id)), labels, indices,
                    unique_indices):
                example = make_example(
                    image=image, label=label, detector_id=detector_id,
                    sample_index=index, unique_index=unique_index
                )
                writer.write(example.SerializeToString())

    # Save the information of each file.
    df = pd.DataFrame(data_info)
    df.to_csv(output_dir / 'data_info.csv')


if __name__ == '__main__':
    main()
