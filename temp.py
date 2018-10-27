import glob
import os.path
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import tensorflow.contrib.slim as slim

# 加载inception_v3模型
import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception_v3

# 导入处理之后的数据文件
INPUT_DATA = r'E:\PythonSpace\finetune_NET\flower_processed_data.npy'
# 定义finetune后变量存储的位置
TRAIN_FILE = r'E:\PythonSpace\finetune_NET\model'
# 预训练的model文件
CKPT_FILE = r'E:\PythonSpace\finetune_NET\inception_v3.ckpt'

# 定义训练中使用的参数
LEARNING_RATE = 0.0001
# 定义训练轮数，每轮训练要跑完所有训练图片
STEPS = 300
# 程序前向运行每次有多少张图片参与
BATCH = 30
# 类别数
N_CLASSES = 5

# finetune时，只是finetune最后的全连接层
CHECKPOINT_EXCLUDE_SCOPES = 'InceptionV3/Logits, InceptionV3/AuxLogits'
TRAINABLE_SCOPES = 'InceptionV3/Logits, InceptionV3/AuxLogits'


# 获取所有需要从训练好的模型中导入数据
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


# 初始化需要训练的两个层的变量
def get_trainable_variables():
    scopes = [scope.strip() for scope in TRAINABLE_SCOPES.split(',')]
    variables_to_train = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    return variables_to_train


def main(argv=None):
    # 加载预处理的数据
    processed_data = np.load(INPUT_DATA)
    training_images = processed_data[0]
    n_training_example = len(training_images)
    training_labels = processed_data[1]
    validation_images = processed_data[2]
    validation_labels = processed_data[3]
    testing_images = processed_data[4]
    testing_labels = processed_data[5]
    print("%d training examples, %d validation examples and %d"
          "testing examples." % (n_training_example, len(validation_labels),
                                 len(testing_labels)))
    # 定义网络的输入
    images = tf.placeholder(tf.float32, [None, 229, 229, 3], name="input_images")
    labels = tf.placeholder(tf.int64, [None], name="labels")
    # 网络的前向运行
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits, _ = inception_v3.inception_v3(images, num_classes=N_CLASSES)

    # 获取需要训练的变量
    trainable_variables = get_trainable_variables()
    # 定义交叉熵损失
    tf.losses.softmax_cross_entropy(tf.one_hot(labels, N_CLASSES), logits, weights=1.0)
    # 优化损失函数
    train_step = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(tf.losses.get_total_loss())

    # 计算正确率
    with tf.name_scope("evaluation"):
        correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 导入预训练好的权重
    load_fn = slim.assign_from_checkpoint_fn(CKPT_FILE, get_tuned_variables(), ignore_missing_vars=True)
    # 用于存储finetune后的权重
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # 初始化没有加载进来的变量
        init = tf.global_variables_initializer()
        sess.run(init)

        print("loading tuned variables from %s" % CKPT_FILE)
        load_fn(sess)

        start = 0
        end = BATCH
        for i in range(STEPS):
            # 开始训练
            sess.run(train_step, feed_dict={
                images: training_images[start:end],
                labels: training_labels[start:end]
            })
            if i % 30 == 0 or i + 1 == STEPS:
                # 这里存储权重时一定要带后面的那个.ckpt
                model_path = os.path.join(TRAIN_FILE, 'model_step' + str(i + 1) + '.ckpt')
                # 保存权重
                saver.save(sess, model_path)
                validation_accuracy = sess.run(evaluation_step, feed_dict={images: validation_images,
                                                                           labels: validation_labels})
                print('Step %d: Validation accuracy = %.1f%%' % (i, validation_accuracy * 100.0))
            start = end
            if start == n_training_example:
                start = 0
            end = start + BATCH
            if end > n_training_example:
                end = n_training_example
        # 训练完成后对测试集进行测试
        test_accuracy = sess.run(evaluation_step, feed_dict={
            images: testing_images, labels: testing_labels})
        print("final test accuracy = %.1f%%" % (test_accuracy * 100))


if __name__ == '__main__':
    tf.app.run()
