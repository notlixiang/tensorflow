from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from datetime import datetime
import os.path
import re
import time

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

import tensorflow.contrib.slim.nets as nets
import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception_v3
from dataset import Dataset, GoodsData
import flags
import image_processing

FLAGS = tf.app.flags.FLAGS

# Constants dictating the learning rate schedule.
RMSPROP_DECAY = 0.9  # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9  # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0  # Epsilon term for RMSProp.

CHECKPOINT_EXCLUDE_SCOPES = 'InceptionV3/Logits, InceptionV3/AuxLogits'
TRAINABLE_SCOPES = 'InceptionV3/Logits, InceptionV3/AuxLogits'


def get_tuned_variables():
    exclusions = [scope.strip() for scope in CHECKPOINT_EXCLUDE_SCOPES.split(',')]

    # 用于存储需要加载参数的名称
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)
    return variables_to_restore


def get_trainable_variables():
    scopes = [scope.strip() for scope in TRAINABLE_SCOPES.split(',')]
    variables_to_train = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    return variables_to_train


def main(_):
    # print(FLAGS.num_preprocess_threads)
    trainset = GoodsData('train')
    assert trainset.data_files()
    validationset = GoodsData('validation')
    assert validationset.data_files()

    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    # get_tuned_variables()
    # get_trainable_variables()

    num_batches_per_epoch = (trainset.num_examples_per_epoch() /
                             FLAGS.batch_size)
    num_preprocess_threads = FLAGS.num_preprocess_threads * FLAGS.num_gpus
    images_train, labels_train = image_processing.distorted_inputs(trainset,
                                                                   num_preprocess_threads=num_preprocess_threads)
    images_validation, labels_validation = image_processing.distorted_inputs(validationset,
                                                                             num_preprocess_threads=num_preprocess_threads)
    # images_splits = tf.split(axis=0, num_or_size_splits=FLAGS.num_gpus, value=images)
    # labels_splits = tf.split(axis=0, num_or_size_splits=FLAGS.num_gpus, value=labels)
    input_summaries = copy.copy(tf.get_collection(tf.GraphKeys.SUMMARIES))
    num_classes = trainset.num_classes() + 1
    print(images_train.shape)
    print(labels_train.shape)
    images = tf.placeholder(tf.float32, [None, images_train.shape[1], images_train.shape[2], 3], name="input_images")
    labels = tf.placeholder(tf.int64, [None], name="labels")
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits, _ = inception_v3.inception_v3(images, num_classes=num_classes)

    # 定义交叉熵损失
    tf.losses.softmax_cross_entropy(tf.one_hot(labels, num_classes), logits, weights=1.0)
    # 优化损失函数
    train_step = tf.train.RMSPropOptimizer(FLAGS.initial_learning_rate).minimize(tf.losses.get_total_loss())
    # 计算正确率
    with tf.name_scope("evaluation"):
        correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    checkpoint_path = tf.train.latest_checkpoint(FLAGS.pretrained_model_checkpoint_path)
    print(get_tuned_variables())
    load_fn = slim.assign_from_checkpoint_fn(checkpoint_path, get_tuned_variables(), ignore_missing_vars=True)

    saver = tf.train.Saver()
    config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=FLAGS.log_device_placement
    )
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)

    # with tf.Session(config=config) as sess:
    # sess.as_default()
    init = tf.global_variables_initializer()
    sess.run(init)

    load_fn(sess)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    batch_size = FLAGS.batch_size
    for step in range(FLAGS.max_steps):
        images_train, labels_train = image_processing.distorted_inputs(trainset,
                                                                       num_preprocess_threads=num_preprocess_threads)
        print(2)
        # labels_train = labels_train.eval(session=sess)
        images_train, labels_train = sess.run([images_train, labels_train])
        print(1)


if __name__ == '__main__':
    tf.app.run()
