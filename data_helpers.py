import numpy as np
import re
import utils
from configure import FLAGS


def word_tokenizer(sentence):
    return sentence.split()


def clean_str(text):
    text = text.lower()
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"that's", "that is ", text)
    text = re.sub(r"there's", "there is ", text)
    text = re.sub(r"it's", "it is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    return text.strip()


def to_one_hot(lst):
    offset = np.array(lst)
    return np.eye(np.max(offset)+1)[offset]


def get_relative_pos(data, max_sentence_length):
    pos1 = []
    pos2 = []
    for idx in range(len(data)):
        tokens = word_tokenizer(data[idx][1])
        e1 = data[idx][2]
        e2 = data[idx][3]
        p1 = ""
        p2 = ""
        # 获得相对实体位置距离的向量
        # 实体所在位置的值为最大索引，前面的值递减 后面的值递增
        for word_idx in range(len(tokens)):
            p1 += str((max_sentence_length-1)+word_idx-e1) + " "
            p2 += str((max_sentence_length-1)+word_idx-e2) + " "
        pos1.append(p1)
        pos2.append(p2)
    return pos1, pos2


def load_data(path):
    print("Loading data...")
    data = []
    lines = [line.strip() for line in open(path)]
    max_sentence_length = 0
    # 遍历每组数据

    # 3        "The <e1>author</e1> of a keygen uses a <e2>disassembler</e2> to look at the raw assembly code."
    # Instrument - Agency(e2, e1)
    # Comment:
    num_data = 0
    for idx in range(0, len(lines), 4):
        num_data = num_data+1
        id = lines[idx].split('\t')[0]
        relation = lines[idx + 1]

        sentence = lines[idx].split('\t')[1]
        sentence = sentence.replace("<e1>", " _e11_ ")
        sentence = sentence.replace("</e1>", " _e12_ ")
        sentence = sentence.replace("<e2>", " _e21_ ")
        sentence = sentence.replace("</e2>", " _e22_ ")
        sentence = clean_str(sentence)

        tokens = word_tokenizer(sentence)
        if max_sentence_length < len(tokens):
            max_sentence_length = len(tokens)
        # 实体1索引
        e1 = tokens.index('e12')-1
        # 实体2索引
        e2 = tokens.index('e22')-1
        data.append([id, sentence, e1, e2, relation])
    print(path)
    print("Max length = {0} , number of data is {1}".format(max_sentence_length, num_data))

    # 得到text
    text = [item[1] for item in data]

    # 得到关系的标签并转为One-hot
    labels = [utils.class2label[item[4]] for item in data]
    # to one-hot
    labels = to_one_hot(labels)
    labels = labels.astype(np.uint8)

    # 获得位置向量
    pos1, pos2 = get_relative_pos(data, FLAGS.max_sentence_length)
    print("Done with loading data.")
    return text, labels, pos1, pos2


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    batch_per_epoch = int((data_size-1)/batch_size)+1
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            data = data[shuffle_indices]
        for batch in range(batch_per_epoch):
            start_index = batch*batch_size
            end_index = min((batch+1)*batch_size, data_size)
            yield [data[start_index:end_index], batch_per_epoch]


if __name__ == '__main__':
    train_file = 'SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT'
    test_file = 'SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT'
    text, labels, pos1, pos2 = load_data(train_file)
    print(labels.shape[1])
    print(len(text[0].split()))








