from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append("E:/proj/tensorflow/models")
sys.path.append("E:/proj/tensorflow/models/research/inception")

import tensorflow as tf


FLAGS = tf.app.flags.FLAGS
image_root_path='E:/proj/17flowers/jpg'

def main(_):
    tf.contrib.data.Dataset.list


if __name__ == '__main__':
  tf.app.run()