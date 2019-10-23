from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pandas as pd
import tensorflow as tf


class Dataset:
    """
    shuffled and repeated/unrepeated dataset
    """

    def __init__(self, csv_path, ds_name=None, per_image_standardization=False,
                 batch_size=16, shuffle_buffer_size=1000,
                 is_repeat=False, is_shuffle=False,
                 input_image_size=(182, 182), target_image_size=None):
        self.input_channels = 4
        csv_path = os.path.expanduser(csv_path)
        assert os.path.exists(csv_path), '%s not found' % csv_path
        # get dataset name
        if ds_name is None:
            self.ds_name = os.path.splitext(os.path.basename(csv_path))[0]
        else:
            self.ds_name = ds_name
        # constants definition
        self.per_image_standardization = per_image_standardization
        assert (target_image_size is None) or (len(target_image_size) == 2), \
            'target_image_size should be set as `None` or a tuple of 2 elements, which means H,W respectively after scaling'
        # target_image_size is None or a tuple/list of 2 elements, meaning W, H respectively
        if target_image_size is None:
            self.target_image_size = input_image_size
        else:
            self.target_image_size = target_image_size
        self.input_image_size = input_image_size

        assert batch_size < shuffle_buffer_size, \
            "Please ensure batch_size < shuffle_buffer_size, " \
            "batch_size: %d, shuffle_buffer_size: %d" % (batch_size, shuffle_buffer_size)

        # read csv file
        df = pd.read_csv(csv_path, header=0, names=['rgb_image_path', 'dep_image_path', 'cls_name'])
        df.cls_name, _ = pd.factorize(df.cls_name, sort=True)

        self._num_of_classes = len(df.cls_name.drop_duplicates())
        self._len_of_dataset = len(df.cls_name)

        image_ds = tf.data.Dataset.from_tensor_slices((df.rgb_image_path.values,
                                                       df.dep_image_path.values))
        label_ds = tf.data.Dataset.from_tensor_slices((df.cls_name,))
        # self.ds = tf.data.Dataset.map(self.ds, self._read_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        image_ds = image_ds.map(lambda x, y: tuple(tf.py_function(self._read_image, [x, y],
                                                                  [tf.float32, ])),
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        image_ds = image_ds.map(self._set_shapes,
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.ds = tf.data.Dataset.zip((image_ds, label_ds))
        if is_shuffle:
            self.ds = self.ds.shuffle(buffer_size=shuffle_buffer_size, seed=2333)
        if is_repeat:
            self.ds = self.ds.repeat()
        self.ds = self.ds.batch(batch_size, drop_remainder=True)
        # self.ds = self.ds.prefetch(tf.data.experimental.AUTOTUNE)

    def _set_shapes(self, image):
        image.set_shape(self.target_image_size + (self.input_channels,))
        return image

    def _read_image(self, rgb_image_path, dep_image_path):

        rgb_image_string = tf.io.read_file(rgb_image_path)
        rgb_image_decoded = tf.image.decode_image(rgb_image_string)
        rgb_image_decoded.set_shape(self.input_image_size + (3,))  # (182, 182, 3)
        rgb_image_convert = tf.cast(rgb_image_decoded, tf.float32)
        dep_image_numpy = np.load(dep_image_path.numpy().decode()).astype(np.float32)

        # dep_image_numpy = tf.py_function(lambda npy_path: np.load(npy_path).astype(np.float32),
        #                                  [dep_image_path], tf.float32)
        dep_image_decoded = tf.convert_to_tensor(dep_image_numpy)
        dep_image_decoded = tf.expand_dims(dep_image_decoded, -1)  # (182, 182, 1)
        dep_image_decoded.set_shape(self.input_image_size + (1,))  # (182, 182, 1)
        dep_image_convert = tf.cast(dep_image_decoded, tf.float32)

        if self.per_image_standardization:
            # mean=0, std=1, uniform distribution
            rgb_image_standard = tf.image.per_image_standardization(rgb_image_convert)
            dep_image_standard = tf.image.per_image_standardization(dep_image_convert)
        else:
            # scaled into [0, 1]
            rgb_image_standard = tf.divide(rgb_image_convert, 255)
            dep_image_standard = tf.divide(dep_image_convert, 255)
        image_standard = tf.concat([rgb_image_standard, dep_image_standard], axis=-1)

        # if self.target_image_size != self.input_image_size:
        #     # image_standard = tf.image.random_crop(image_standard,
        #     #                                       size=self.target_image_size + (self.input_channels, ))
        #     image_standard = tf.image.resize(image_standard,
        #                                      size=self.target_image_size)
        # label_vector = tf.one_hot(label_id, self.num_of_classes)
        return image_standard

    def get_num_of_classes(self):
        return self._num_of_classes

    def __len__(self):
        return self._len_of_dataset


if __name__ == '__main__':
    dataset = Dataset('~/vggface3d_sm/train.csv', target_image_size=(160, 160), batch_size=8, is_shuffle=True)
    print(tf.compat.v1.data.get_output_shapes(dataset.ds))
    print(dataset.get_num_of_classes())
    i = 0
    # import time
    # start = time.time()
    for img, cls in dataset.ds:
        print(i, img, cls)
        i = i + 1
        if i >= 1:
            break
    # elapsed = (time.time() - start)
    # print("elapsed time: %d" % elapsed)
