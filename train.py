import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import os
import time
import json

import utils
import data_helpers
from configure import FLAGS
from model import *

from sklearn.metrics import f1_score
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)


def train():
    with tf.device('/cpu:0'):
        x_text, y, pos1, pos2 = data_helpers.load_data(FLAGS.train_path)

    # 建立词映射表
    # Example: x_text[k] = 'the e11 factory e12 products have included flower pots finnish rooster'
    # =>[1  2  4  3  5  6  7  8  9 10 11]
    # =>[1 2 4 3 5 6 7 8 9 10 11 0  0 ... 0 0]

    text_tokenizer = keras.preprocessing.text.Tokenizer()
    text_tokenizer.fit_on_texts(x_text)
    x_text = text_tokenizer.texts_to_sequences(x_text)
    x = keras.preprocessing.sequence.pad_sequences(x_text, FLAGS.max_sentence_length, padding='post')

    text_vocab_size = len(text_tokenizer.word_index)
    print("Text vocabulary size:{}".format(text_vocab_size))
    print("x shape={0}".format(x.shape))
    print("y shape={0}".format(y.shape))
    print("")

    # 建立位置向量
    # pos1[k] = ['32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55']
    # => [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
    #    0  0  0  0  0  0  0  0  0  0  0]
    pos_tokenizer = keras.preprocessing.text.Tokenizer()
    pos_tokenizer.fit_on_texts(pos1+pos2)
    p1 = pos_tokenizer.texts_to_sequences(pos1)
    p2 = pos_tokenizer.texts_to_sequences(pos2)
    p1 = keras.preprocessing.sequence.pad_sequences(p1, FLAGS.max_sentence_length, padding='post')
    p2 = keras.preprocessing.sequence.pad_sequences(p2, FLAGS.max_sentence_length, padding='post')

    pos_vocab_size = len(pos_tokenizer.word_index)
    print("Position vocabulary size:{}".format(pos_vocab_size))
    print("pos_1 shape={0}".format(p1.shape))
    print("pos_2 shape={0}".format(p2.shape))
    print("")

    # 随机打乱数据然后分为训练和测试数据
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]
    p1_shuffled = p1[shuffle_indices]
    p2_shuffled = p2[shuffle_indices]
    dev_sample_index = -1*int(float(len(y))*FLAGS.dev_sample_percentage)
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    p1_train, p1_dev = p1_shuffled[:dev_sample_index], p1_shuffled[dev_sample_index:]
    p2_train, p2_dev = p2_shuffled[:dev_sample_index], p2_shuffled[dev_sample_index:]
    print("Train/Dev split:{0}/{1}".format(len(y_train), len(y_dev)))
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = FLAGS.gpu_allow_growth
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                x.shape[1],
                y.shape[1],
                text_vocab_size+1,
                pos_vocab_size+1,
                FLAGS.text_embedding_dim,
                FLAGS.pos_embedding_dim,
                list(map(int, FLAGS.filter_sizes.split(","))),
                FLAGS.num_filters,
                FLAGS.l2_reg_lambda
            )

            # 定义训练步骤
            global_step = tf.Variable(0, trainable=False, name='global_step')
            optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
            # optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
            # op.minimize()的第一步 拆开以梯度修剪
            gvs = optimizer.compute_gradients(cnn.loss)
            # 梯度修剪
            capped_gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gvs]
            train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)

            # 输出路径
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            # 记录
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            accuracy_summary = tf.summary.scalar("accuracy", cnn.accuracy)
            # 训练记录
            train_summary_op = tf.summary.merge([loss_summary, accuracy_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
            # 验证记录
            dev_summary_op = tf.summary.merge([loss_summary, accuracy_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)
            # checkoutpoint 输出
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # 保存文本和位置的映射表
            with open(os.path.join(out_dir, 'text_tokenizer.json'), 'w') as js:
                json.dump(text_tokenizer.word_index, js)
            with open(os.path.join(out_dir, 'pos_tokenizer.json'), 'w') as js:
                json.dump(pos_tokenizer.word_index, js)

            # 初始化所有参数
            sess.run(tf.global_variables_initializer())

            # 预训练
            if FLAGS.embedding_path:
                Pretrained_W = utils.load_word2vec(FLAGS.embedding_path, FLAGS.text_embedding_dim, text_tokenizer)
                sess.run(cnn.W_text.assign(Pretrained_W))
                print("Load Pretrained Embedding Success!")
            # 生成batch训练数据
            data = list(zip(x_train, p1_train, p2_train, y_train))
            batches = data_helpers.batch_iter(data, FLAGS.batch_size, FLAGS.num_epochs, True)
            best_f1 = 0.0
            cnt_epoch = 0
            cnt_batch = 0
            for batch_and_per_batches in batches:
                batches_per_epoch = batch_and_per_batches[1]
                batch = batch_and_per_batches[0]
                cnt_batch = cnt_batch+1
                x_batch, p1_batch, p2_batch, y_batch = zip(*batch)
                feed_dic = {
                    cnn.input_text: x_batch,
                    cnn.input_p1: p1_batch,
                    cnn.input_p2: p2_batch,
                    cnn.input_y: y_batch,
                    cnn.drop_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict=feed_dic
                )
                train_summary_writer.add_summary(summaries, step)
                if cnt_batch == batches_per_epoch:
                    cnt_epoch = cnt_epoch+1
                    feed_dict = {
                        cnn.input_text: x_dev,
                        cnn.input_p1: p1_dev,
                        cnn.input_p2: p2_dev,
                        cnn.input_y: y_dev,
                        cnn.drop_keep_prob: 1.0
                    }
                    summaries, loss, accuracy, predictions = sess.run(
                        [dev_summary_op, cnn.loss, cnn.accuracy, cnn.predictions], feed_dict)
                    dev_summary_writer.add_summary(summaries, step)

                    f1 = f1_score(np.argmax(y_dev, 1), predictions, labels=np.array(range(1, 19)), average='macro')
                    print("epoch {0} --- loss: {1}  acc:{2}  f1:{3}".format(cnt_epoch, loss, accuracy, f1))
                    if best_f1 < f1:
                        best_f1 = f1
                        path = saver.save(sess, checkpoint_prefix+"-{:.3f}".format(best_f1), global_step=step)
                        print("Model saved to {}".format(path))
                    cnt_batch = 0


def main(_):
    train()


if __name__ == '__main__':
    tf.app.run()
