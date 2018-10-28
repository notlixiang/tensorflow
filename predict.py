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
from tensorflow import GraphKeys

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


def get_all_variables():
    scopes = [scope.strip() for scope in TRAINABLE_SCOPES.split(',')]
    variables_to_train = []
    # for scope in scopes:
    #     variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
    #     variables_to_train.extend(variables)
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    variables_to_train.extend(variables)
    return variables_to_train

def main(_):
    print(FLAGS.num_preprocess_threads)
    trainset = GoodsData('train')
    # assert trainset.data_files()
    validationset = GoodsData('validation')
    assert validationset.data_files()

    # get_tuned_variables()
    # get_trainable_variables()

    # num_batches_per_epoch = (trainset.num_examples_per_epoch() /
    #                          FLAGS.batch_size)
    num_preprocess_threads = FLAGS.num_preprocess_threads * FLAGS.num_gpus
    images_train, labels_train = image_processing.distorted_inputs(trainset,
                                                                   num_preprocess_threads=num_preprocess_threads)
    images_validation, labels_validation = image_processing.distorted_inputs(validationset,batch_size=64,
                                                                   num_preprocess_threads=num_preprocess_threads)
    # images_splits = tf.split(axis=0, num_or_size_splits=FLAGS.num_gpus, value=images)
    # labels_splits = tf.split(axis=0, num_or_size_splits=FLAGS.num_gpus, value=labels)

    input_summaries = copy.copy(tf.get_collection(tf.GraphKeys.SUMMARIES))

    # Number of classes in the Dataset label set plus 1.
    # Label 0 is reserved for an (unused) background class.
    num_classes = trainset.num_classes() + 1
    # print(images_train.shape)
    # print(labels_train.shape)
    images = tf.placeholder(tf.float32, [None, images_train.shape[1], images_train.shape[2], 3], name="input_images")
    labels = tf.placeholder(tf.int64, [None], name="labels")
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits, _ = inception_v3.inception_v3(images, num_classes=num_classes)


    tuned_variables = get_all_variables()
    trainable_variables = get_all_variables()
    checkpoint_path = FLAGS.pretrained_model_checkpoint_path

    # 计算正确率
    with tf.name_scope("evaluation"):
        prediction=tf.argmax(logits, 1)
        correct_prediction = tf.equal(prediction, labels)
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 导入预训练好的权重
    checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
    load_fn = slim.assign_from_checkpoint_fn(checkpoint_path, tuned_variables, ignore_missing_vars=True)
    # 用于存储finetune后的权重
    # print(get_tuned_variables())
    # saver = tf.train.Saver()


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

    print("loading tuned variables from %s" % checkpoint_path)
    load_fn(sess)
    # sess.run(load_fn)
    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(coord=coord)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # tf.train.batch
    # start = 0
    # end = FLAGS.batch_size

    # if tf.gfile.Exists(FLAGS.train_dir):
    #     tf.gfile.DeleteRecursively(FLAGS.train_dir)
    # tf.gfile.MakeDirs(FLAGS.train_dir)

    for step in range(FLAGS.max_steps):
        # print(0)
        start_time = time.time()

        image_batch, label_batch = sess.run([images_validation, labels_validation])
        validation_accuracy = sess.run(evaluation_step, feed_dict={images: image_batch,
                                                                   labels: label_batch})
        label_prediction=sess.run(prediction,feed_dict={images: image_batch,
                                                                   labels: label_batch})
        print(label_prediction)
        print('Step %d: Validation accuracy = %.1f%%' % (step, validation_accuracy * 100.0))
        duration = time.time() - start_time


if __name__ == '__main__':
    tf.app.run()
