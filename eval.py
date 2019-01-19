import tensorflow as tf
import numpy as np
import json
import os
from sklearn.metrics import f1_score

import data_helpers
import utils
from configure import FLAGS


class Tokenizer:
    def __init__(self, path):
        self.word_index = {}
        with open(path, 'r') as f:
            self.word_index = json.load(f)

    def transform(self, text, max_sequence_length):
        sequences = []
        for i in range(len(text)):
            sequence = []
            tokens = text[i].split()
            for j in range(len(tokens)):
                if tokens[j] in self.word_index:
                    sequence.append(self.word_index[tokens[j]])
                else:
                    sequence.append(0)
            j = len(tokens)
            while j < max_sequence_length:
                sequence.append(0)
                j = j+1
            sequences.append(sequence)
        return sequences


def eval():
    with tf.device('/cpu:0'):
        x_text, y, pos1, pos2 = data_helpers.load_data(FLAGS.test_path)

    # 读取词汇表,并转为id表示的数据
    text_tokenizer = Tokenizer(FLAGS.text_tokenizer_path)
    pos_tokenizer = Tokenizer(FLAGS.pos_tokenizer_path)
    x = np.array(text_tokenizer.transform(x_text, FLAGS.max_sentence_length), dtype=np.int32)
    p1 = np.array(pos_tokenizer.transform(pos1, FLAGS.max_sentence_length), dtype=np.int32)
    p2 = np.array(pos_tokenizer.transform(pos2, FLAGS.max_sentence_length), dtype=np.int32)
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = FLAGS.gpu_allow_growth
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # 读取计算图和会话
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # 获得placeholder
            input_text = graph.get_operation_by_name("input_text").outputs[0]
            input_p1 = graph.get_operation_by_name("input_p1").outputs[0]
            input_p2 = graph.get_operation_by_name("input_p2").outputs[0]
            drop_keep_prob = graph.get_operation_by_name("keep_prob").outputs[0]

            predictions = graph.get_operation_by_name("out_put/predictions").outputs[0]

            # 生成batch数据
            batches = data_helpers.batch_iter(list(zip(x, p1, p2)), FLAGS.batch_size, 1, False)

            # 预测
            preds = []
            for batch_and_num_batches in batches:
                batch = batch_and_num_batches[0]
                x_batch, p1_batch, p2_batch = zip(*batch)
                pred = sess.run(predictions, feed_dict={
                    input_text: x_batch,
                    input_p1: p1_batch,
                    input_p2: p2_batch,
                    drop_keep_prob: 1.0
                })
                preds.append(pred)
            preds = np.concatenate(preds).astype(np.uint8)
            labels = np.argmax(y, 1)

            # 评价指标
            correct_vector = np.equal(preds, labels).astype(np.uint8)
            accuracy = np.mean(correct_vector)
            f1 = f1_score(labels, preds, labels=np.array(range(1, 19)), average='macro')
            print("Evalation Result:\nAccuracy:{}\nF1:{}".format(accuracy, f1))

            # 记录预测结果
            prediction_path = os.path.join(FLAGS.checkpoint_dir, "..", "predictions.txt")
            prediction_file = open(prediction_path, 'w')
            prediction_file.write("Evalation Result:\nAccuracy:{}\nF1:{}\n\n".format(accuracy, f1))
            prediction_file.write("ID\tPrediction\tGround Truth\n")
            for i in range(len(preds)):
                prediction_file.write("{}\t{}\t{}\n".format(i, utils.label2class[preds[i]],
                                                            utils.label2class[labels[i]]))
            prediction_file.close()


def main(_):
    eval()


if __name__ == "__main__":
    tf.app.run()