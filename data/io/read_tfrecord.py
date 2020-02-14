# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf
import os
from data.io import image_preprocess
from libs.configs import cfgs


features={
    'img_name': tf.FixedLenFeature([], tf.string),
    'img_height': tf.FixedLenFeature([], tf.int64),
    'img_width': tf.FixedLenFeature([], tf.int64),
    'img': tf.FixedLenFeature([], tf.string),
    'gtboxes_and_label': tf.FixedLenFeature([], tf.string),
    'num_objects': tf.FixedLenFeature([], tf.int64)
}


def _parse_image_function(example_proto):
    return tf.parse_single_example(example_proto, features)


def read_single_example_and_decode(raw_dataset):

    # tfrecord_options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)

    # reader = tf.TFRecordReader(options=tfrecord_options)
    # reader = tf.TFRecordReader()
    # _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized=raw_dataset,
        features={
            'img_name': tf.FixedLenFeature([], tf.string),
            'img_height': tf.FixedLenFeature([], tf.int64),
            'img_width': tf.FixedLenFeature([], tf.int64),
            'img': tf.FixedLenFeature([], tf.string),
            'gtboxes_and_label': tf.FixedLenFeature([], tf.string),
            'num_objects': tf.FixedLenFeature([], tf.int64)
        }
    )
    img_name = features['img_name']
    img_height = tf.cast(features['img_height'], tf.int32)
    img_width = tf.cast(features['img_width'], tf.int32)
    img = tf.decode_raw(features['img'], tf.uint8)

    img = tf.reshape(img, shape=[img_height, img_width, 3])
    # DOTA dataset need exchange img_width and img_height
    # img = tf.reshape(img, shape=[img_width, img_height, 3])

    gtboxes_and_label = tf.decode_raw(features['gtboxes_and_label'], tf.int32)
    gtboxes_and_label = tf.reshape(gtboxes_and_label, [-1, 9])

    num_objects = tf.cast(features['num_objects'], tf.int32)
    return img_name, img, gtboxes_and_label, num_objects


def read_and_prepocess_single_img(raw_dataset, shortside_len, is_training):

    img_name, img, gtboxes_and_label, num_objects = read_single_example_and_decode(raw_dataset)

    img = tf.cast(img, tf.float32)
    img = img - tf.constant(cfgs.PIXEL_MEAN)
    if is_training:
        img, gtboxes_and_label = image_preprocess.short_side_resize(img_tensor=img, gtboxes_and_label=gtboxes_and_label,
                                                                    target_shortside_len=shortside_len)
        img, gtboxes_and_label = image_preprocess.random_flip_left_right(img_tensor=img,
                                                                         gtboxes_and_label=gtboxes_and_label)

    else:
        img, gtboxes_and_label = image_preprocess.short_side_resize(img_tensor=img, gtboxes_and_label=gtboxes_and_label,
                                                                    target_shortside_len=shortside_len)

    return img_name, img, gtboxes_and_label, num_objects


def next_batch(dataset_name, batch_size, shortside_len, is_training):
    '''
    :return:
    img_name_batch: shape(1, 1)
    img_batch: shape:(1, new_imgH, new_imgW, C)
    gtboxes_and_label_batch: shape(1, Num_Of_objects, 5] .each row is [x1, y1, x2, y2, label]
    '''
    assert batch_size == 1, "we only support batch_size is 1.We may support large batch_size in the future"

    if dataset_name not in ['jyzdata', 'DOTA', 'ship', 'ICDAR2015', 'pascal', 'coco', 'DOTA_TOTAL', 'WIDER']:
        raise ValueError('dataSet name must be in pascal, coco spacenet and ship')

    if is_training:
        pattern = os.path.join('/home/work/tfrecord/', dataset_name + '_val.tfrecord')
    else:
        pattern = os.path.join('/content/drive/', dataset_name + '_test.tfrecord')

    print('tfrecord path is -->', os.path.abspath(pattern))

    # filename_tensorlist = tf.train.match_filenames_once(pattern)

    # filename_queue = tf.train.string_input_producer([pattern])
    raw_dataset = tf.data.TFRecordDataset(pattern)
    raw_dataset = raw_dataset.map(_parse_image_function)
    raw_dataset = raw_dataset.repeat()
    raw_dataset = raw_dataset.shuffle(1000)
    raw_dataset = tf.data.make_one_shot_iterator(raw_dataset)
    # parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
    # raw_dataset = tf.python_io.tf_record_iterator(path=pattern)

    shortside_len = tf.constant(shortside_len)
    shortside_len = tf.random_shuffle(shortside_len)[0]

    single_ex = raw_dataset.get_next()

    img_name = single_ex['img_name']
    img_height = tf.cast(single_ex['img_height'], tf.int32)
    img_width = tf.cast(single_ex['img_width'], tf.int32)
    img = tf.decode_raw(single_ex['img'], tf.uint8)

    img = tf.reshape(img, shape=[img_height, img_width, 3])

    gtboxes_and_label = tf.decode_raw(single_ex['gtboxes_and_label'], tf.int32)
    gtboxes_and_label = tf.reshape(gtboxes_and_label, [-1, 9])

    num_obs = tf.cast(single_ex['num_objects'], tf.int32)

    img = tf.cast(img, tf.float32)
    img = img - tf.constant(cfgs.PIXEL_MEAN)
    if is_training:
        img, gtboxes_and_label = image_preprocess.short_side_resize(img_tensor=img, gtboxes_and_label=gtboxes_and_label,
                                                                    target_shortside_len=shortside_len)
        img, gtboxes_and_label = image_preprocess.random_flip_left_right(img_tensor=img,
                                                                         gtboxes_and_label=gtboxes_and_label)

    else:
        img, gtboxes_and_label = image_preprocess.short_side_resize(img_tensor=img, gtboxes_and_label=gtboxes_and_label,
                                                                    target_shortside_len=shortside_len)


    # img_name, img, gtboxes_and_label, num_obs = read_and_prepocess_single_img(raw_dataset, shortside_len,
    #                                                                           is_training=is_training)
    img_name_batch, img_batch, gtboxes_and_label_batch , num_obs_batch = \
        tf.train.batch(
                       [img_name, tf.to_float(img), gtboxes_and_label, num_obs],
                       batch_size=batch_size,
                       capacity=1,
                       num_threads=1,
                       dynamic_pad=True)
    return img_name_batch, img_batch, gtboxes_and_label_batch, num_obs_batch


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    img_name_batch, img_batch, gtboxes_and_label_batch, num_objects_batch = \
        next_batch(dataset_name=cfgs.DATASET_NAME,  # 'pascal', 'coco'
                   batch_size=cfgs.BATCH_SIZE,
                   shortside_len=cfgs.IMG_SHORT_SIDE_LEN,
                   is_training=True)
    gtboxes_and_label = tf.reshape(gtboxes_and_label_batch, [-1, 9])

    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)

        img_name_batch_, img_batch_, gtboxes_and_label_batch_, num_objects_batch_ \
            = sess.run([img_name_batch, img_batch, gtboxes_and_label_batch, num_objects_batch])

        print('debug')

        coord.request_stop()
        coord.join(threads)
