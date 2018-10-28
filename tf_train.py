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


def get_all_variables():
    scopes = [scope.strip() for scope in TRAINABLE_SCOPES.split(',')]
    variables_to_train = []
    # for scope in scopes:
    #     variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
    #     variables_to_train.extend(variables)
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    variables_to_train.extend(variables)
    return variables_to_train


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

    # get_tuned_variables()
    # get_trainable_variables()

    num_batches_per_epoch = (trainset.num_examples_per_epoch() /
                             FLAGS.batch_size)
    num_preprocess_threads = FLAGS.num_preprocess_threads * FLAGS.num_gpus

    if FLAGS.from_official == True:
        batch_size = FLAGS.batch_size * 4
    else:
        batch_size = FLAGS.batch_size

    images_train, labels_train = image_processing.distorted_inputs(trainset,
                                                                   batch_size=batch_size,
                                                                   num_preprocess_threads=num_preprocess_threads)
    images_validation, labels_validation = image_processing.distorted_inputs(validationset, batch_size=64,
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

    if FLAGS.from_official == True:
        tuned_variables = get_tuned_variables()
        trainable_variables = get_trainable_variables()
        checkpoint_path = FLAGS.official_checkpoint_path
    else:
        tuned_variables = get_all_variables()
        trainable_variables = get_all_variables()
        checkpoint_path = FLAGS.pretrained_model_checkpoint_path

    # print(trainable_variables)
    # test_tafafdsa=GraphKeys.TRAINABLE_VARIABLES
    # 获取需要训练的变量
    # trainable_variables = get_trainable_variables()
    # 定义交叉熵损失
    # 优化损失函数
    tf.losses.softmax_cross_entropy(tf.one_hot(labels, num_classes), logits, weights=1.0)
    optimizer = tf.train.AdamOptimizer()
    loss = tf.losses.get_total_loss()
    train_step = optimizer.minimize(loss, var_list=trainable_variables)

    # total_loss=tf.losses.softmax_cross_entropy(tf.one_hot(labels, num_classes), logits, weights=1.0)
    # train_step = tf.train.RMSPropOptimizer(FLAGS.initial_learning_rate).minimize(total_loss)

    # 计算正确率
    with tf.name_scope("evaluation"):
        correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 导入预训练好的权重
    checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
    load_fn = slim.assign_from_checkpoint_fn(checkpoint_path, tuned_variables, ignore_missing_vars=True)
    # 用于存储finetune后的权重
    # print(get_tuned_variables())
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

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs/", sess.graph)

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

    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
    saver.save(sess, checkpoint_path, global_step=0)
    for step in range(FLAGS.max_steps):
        # print(0)
        start_time = time.time()
        # print(1)
        # image_batch = sess.run(images_train[start:end])
        # print(2)
        # # label_batch = sess.run(labels_train[start:end])
        # label_batch = labels_train[start:end]
        #
        # print(3)
        # images_train, labels_train = image_processing.distorted_inputs(trainset,
        #                                                                num_preprocess_threads=num_preprocess_threads)
        # images_validation, labels_validation = image_processing.distorted_inputs(validationset,
        #                                                                          num_preprocess_threads=num_preprocess_threads)

        image_batch, label_batch = sess.run([images_train, labels_train])
        # print(3)
        # sess.run(train_step, feed_dict={
        #     images: image_batch,
        #     labels: label_batch
        # })
        # print(4)
        # print(1)
        # print(label_batch)
        # loss_tensor = tf.losses.get_total_loss()
        sess.run(train_step, feed_dict={images: image_batch, labels: label_batch})
        # loss_now=sess.run(loss)
        # print(2)
        duration = time.time() - start_time

        # assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        if step % 5 == 0:
            examples_per_sec = FLAGS.batch_size / float(duration)
            format_str = ('%s: step %d,'  # loss = %.2f 
                          '(%.1f examples/sec; %.3f '
                          'sec/batch)')
            print(format_str % (datetime.now(), step,  # loss_now,
                                examples_per_sec, duration))
        if step % 50 == 0:
            image_batch, label_batch = sess.run([images_validation, labels_validation])
            validation_accuracy = sess.run(evaluation_step, feed_dict={images: image_batch,
                                                                       labels: label_batch})
            result = sess.run(merged, feed_dict={images: image_batch,
                                                 labels: label_batch})
            writer.add_summary(result, step)
            print('Step %d: Validation accuracy = %.1f%%' % (step, validation_accuracy * 100.0))

        # Save the model checkpoint periodically.
        if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
            checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)


if __name__ == '__main__':
    tf.app.run()
