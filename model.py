import tensorflow as tf


class TextCNN:
    def __init__(self, sequence_length, num_classes, text_vocab_size, pos_vocab_size,
                 text_embedding_size, pos_embedding_size, filter_sizes, num_filters,
                 l2_reg_lambda=0.0):

        # 输入输出的占位符
        self.input_text = tf.placeholder(dtype=tf.int32, shape=[None, sequence_length], name="input_text")
        self.input_p1 = tf.placeholder(dtype=tf.int32, shape=[None, sequence_length], name="input_p1")
        self.input_p2 = tf.placeholder(dtype=tf.int32, shape=[None, sequence_length], name="input_p2")
        self.input_y = tf.placeholder(dtype=tf.float32, shape=[None, num_classes], name="input_y")
        self.drop_keep_prob = tf.placeholder(dtype=tf.float32, name="keep_prob")

        initializer = tf.initializers.glorot_normal

        # Embedding 层
        with tf.device('/cpu:0'), tf.variable_scope("text_embedding"):
            self.W_text = tf.Variable(tf.random_uniform([text_vocab_size, text_embedding_size], -0.25, 0.25),
                                      name="W_text")
            self.text_embedding_chars = tf.nn.embedding_lookup(self.W_text, self.input_text)
            # 这里扩充一个维度 是作为channel通道
            self.text_embedding_chars_expand = tf.expand_dims(self.text_embedding_chars, -1)

        with tf.device('/cpu:0'), tf.variable_scope("pos_embedding"):
            self.W_pos = tf.get_variable("W_pos", [pos_vocab_size, pos_embedding_size],
                                         initializer=initializer())
            self.p1_embedding_chars = tf.nn.embedding_lookup(self.W_pos, self.input_p1)
            self.p1_embedding_chars_expand = tf.expand_dims(self.p1_embedding_chars, -1)
            self.p2_embedding_chars = tf.nn.embedding_lookup(self.W_pos, self.input_p2)
            self.p2_embedding_chars_expand = tf.expand_dims(self.p2_embedding_chars, -1)

        # 把一个单词的三个embedding拼在一起
        self.embedding_chars = tf.concat([self.text_embedding_chars_expand,
                                          self.p1_embedding_chars_expand,
                                          self.p2_embedding_chars_expand], 2)
        self.embedding_size = text_embedding_size+2*pos_embedding_size

        # 卷积和池化层
        pool_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.variable_scope("conv-maxpool-%s" % filter_size):
                # NxSxEx1
                conv = tf.layers.conv2d(self.embedding_chars, num_filters,
                                        [filter_size, self.embedding_size],
                                        kernel_initializer=initializer(), activation=tf.nn.relu, name='conv')
                # NxS'x1xCi
                pool = tf.nn.max_pool(conv, ksize=[1, sequence_length-filter_size+1, 1, 1],
                                      strides=[1, 1, 1, 1], padding='VALID', name='pool')
                # Nx1x1xCi
                # N为数据个数 S为句子长度 E为 Embdedding 长度 Ci为通道数
                pool_outputs.append(pool)

        # 把所有不同size的filter的池化结果拼起来
        num_filters_total = num_filters*len(filter_sizes)
        # 把通道维拼起来
        self.h_pool = tf.concat(pool_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # dropout层
        with tf.variable_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, keep_prob=self.drop_keep_prob)

        # 最终输出
        with tf.variable_scope("out_put"):
            self.full_connected = tf.layers.dense(self.h_drop, num_classes, kernel_initializer=initializer())
            self.predictions = tf.argmax(self.full_connected, 1, name='predictions')

        # 损失函数
        with tf.variable_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.full_connected, labels=self.input_y)
            self.l2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda*self.l2

        # 精度
        with tf.variable_scope("accuracy"):
            correct_vector = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_vector, dtype=tf.float32), name='accuracy')


