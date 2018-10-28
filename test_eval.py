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
import build_image_data

FLAGS = tf.app.flags.FLAGS

# Constants dictating the learning rate schedule.
RMSPROP_DECAY = 0.9  # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9  # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0  # Epsilon term for RMSProp.

CHECKPOINT_EXCLUDE_SCOPES = 'InceptionV3/Logits, InceptionV3/AuxLogits'
TRAINABLE_SCOPES = 'InceptionV3/Logits, InceptionV3/AuxLogits'


import logging
import random
import time

from flask import Flask, jsonify, request

import numpy as np

app = Flask(__name__)


def get_all_variables():
    scopes = [scope.strip() for scope in TRAINABLE_SCOPES.split(',')]
    variables_to_train = []
    # for scope in scopes:
    #     variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
    #     variables_to_train.extend(variables)
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    variables_to_train.extend(variables)
    return variables_to_train

def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
                                input_mean=128, input_std=128):
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(file_reader, channels=3,
                                           name='png_reader')
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                      name='gif_reader'))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
    else:
        image_reader = tf.image.decode_jpeg(file_reader, channels=3,
                                            name='jpeg_reader')
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0);
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)

    return result


def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  label.append('unknown')
  return label


@app.route('/')
def classify():
    file_name = request.args['file']

    img_recv = read_tensor_from_image_file(file_name,
                                    input_height=FLAGS.image_size,
                                    input_width=FLAGS.image_size,)

    # logits_tensor = sess.graph.get_tensor_by_name('logits')
    start = time.time()
    results_softmax=tf.nn.softmax(logits=logits)
    results=sess.run(results_softmax,feed_dict={images: img_recv})
    # results = sess.run(output_operation.outputs[0],
    #                    {input_operation.outputs[0]: t})
    end = time.time()
    results = np.squeeze(results)

    top_k = results.argsort()[-3:][::-1]

    print('\nEvaluation time (1-image): {:.3f}s\n'.format(end - start))

    for i in top_k:
        print(lables_output[i], results[i])

    return jsonify(lables_output, results.tolist())

# def main(_):


if __name__ == '__main__':
    # print(FLAGS.num_preprocess_threads)
    model_file = "retrained_graph.pb"
    label_file = "retrained_labels.txt"
    trainset = GoodsData('train')
    assert trainset.data_files()
    validationset = GoodsData('validation')
    assert validationset.data_files()

    lables_output = load_labels(FLAGS.labels_file)

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
        logits, endpoints = inception_v3.inception_v3(images, num_classes=num_classes)

    # 定义交叉熵损失
    # tf.losses.softmax_cross_entropy(tf.one_hot(labels, num_classes), logits, weights=1.0)
    # 优化损失函数
    # train_step = tf.train.RMSPropOptimizer(FLAGS.initial_learning_rate).minimize(tf.losses.get_total_loss())
    # 计算正确率
    with tf.name_scope("evaluation"):
        correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    checkpoint_path = tf.train.latest_checkpoint(FLAGS.pretrained_model_checkpoint_path)
    # print(get_tuned_variables())
    load_fn = slim.assign_from_checkpoint_fn(checkpoint_path, get_all_variables(), ignore_missing_vars=True)

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


    # for step in range(FLAGS.max_steps):
    #     images_train, labels_train = image_processing.distorted_inputs(trainset,
    #                                                                    num_preprocess_threads=num_preprocess_threads)
    #     print(2)
    #     # labels_train = labels_train.eval(session=sess)
    #     images_train, labels_train = sess.run([images_train, labels_train])
    #     print(1)
    app.run(debug=True, port=8000)