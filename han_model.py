import tensorflow as tf
from tensorflow.contrib import rnn
from sklearn.metrics import classification_report


class HierarchicalAttention(object):
    def __init__(self, config, embedding, initializer=tf.random_normal_initializer(stddev=0.1)):
        self.config = config
        self.hidden_size = self.config.hidden_size
        self.gru_output_keep_prob = self.config.gru_output_keep_prob
        self.initializer = initializer  # return Gaussian distribution initializer tensor
        self.embedding = embedding
        # self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate * 0.5)
        self.input_x = tf.placeholder(tf.int32, [None, self.config.sequence_length], name='input_x')
        self.sequence_length = int(self.config.sequence_length / self.config.num_sentence)
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')
        self.input_y = tf.placeholder(tf.int32, [None, self.config.class_num], name='input_y')
        self.init_weight()
        self.logits = self.inference()
        self.loss = self.classficatoin_text_loss(self.logits)
        self.optim = self.classficatoin_text_train(self.loss)
        self.accuracy = self.classficatoin_text_accuarcy(self.logits)

    def inference(self):
        """
        start HAN
        :return: logits
        """
        # convert to embedding
        with tf.name_scope('word_embedding'):
            input_x = tf.split(self.input_x, self.config.num_sentence, axis=1)
            input_x = tf.stack(input_x, axis=1)

            input_x = tf.nn.embedding_lookup(self.embedding, input_x)
            input_x = tf.reshape(input_x, [-1, self.sequence_length, self.hidden_size])

        with tf.name_scope('word_forward'):
            hidden_state_forward_word, _ = self.gru_forward(input_x, self.batch_size * self.config.num_sentence,
                                                            self.config.hidden_size, "word_forward")
        with tf.name_scope('word_backward'):
            hidden_state_backward_word, _ = self.gru_backward(input_x, self.batch_size * self.config.num_sentence,
                                                              self.config.hidden_size, "word_backward")

        """
            concat forwards and backwards output,its hidden size will be 2*hidden_size
        """
        with tf.name_scope('word_attention'):
            hidden_state_word = tf.concat([hidden_state_forward_word, hidden_state_backward_word], axis=2)
            # Word Attention
            word_representation = self.word_attention(hidden_state_word)
            word_representation = tf.reshape(word_representation, shape=[-1, self.config.num_sentence,
                                                                         self.hidden_size * 2])
        # Sentence Attention
        with tf.name_scope('sentence_forward'):
            hidden_state_forward_sentences, _ = self.gru_forward(word_representation, self.batch_size,
                                                                 self.hidden_size * 2, "sentence_forward")
        with tf.name_scope('sentence_backward'):
            hidden_state_backward_sentences, _ = self.gru_backward(word_representation, self.batch_size,
                                                                   self.hidden_size * 2, "sentence_backward")

        """
            concat forwards and backwards output,its hidden size will be 4*hidden_size
        """
        with tf.name_scope('sentence_attention'):
            hidden_state_sentence = tf.concat([hidden_state_forward_sentences, hidden_state_backward_sentences],
                                              axis=2)
            document_representation = self.sentence_attention(hidden_state_sentence)

        logits = self.classficatoin_text_logits(document_representation)

        return logits

    def gru_forward(self, input_x, zero_state_length, hidden_size, name_variable):
        """
        GRU forward
        :param input_x:shape: [batch_size*num_sentence,sequence_length,embedding_size]
        :param zero_state_length: gre cell zero state size
        :param hidden_size: gru output hidden size
        :param name_variable: name of gru variable
        :return: GRU forward outputs and every time step state
        """
        with tf.variable_scope(name_variable):
            gru_cell = self.create_gru_unit(hidden_size)

            # init unit state, this is able to init gru state ,each of data of batch need to be initializer, when train
            gru_init_state = gru_cell.zero_state(zero_state_length, dtype=tf.float32)
            outputs, state = tf.nn.dynamic_rnn(gru_cell, inputs=input_x, initial_state=gru_init_state)

        return outputs, state

    def gru_backward(self, input_x, zero_state_length, hidden_size, name_variable):
        """
        GRU backward
        :param input_x:shape:[None*num_sentence, sequence_length, embedding_size]
        :param zero_state_length: gre cell zero state size
        :param hidden_size: gre output hidden size
        :param name_variable: name of gru variable
        :return:GRU backward outputs and every time step state
        """
        with tf.variable_scope(name_variable):
            input_x = tf.reverse_v2(input_x, axis=[1])
            gru_cell = self.create_gru_unit(hidden_size)

            # init unit state
            gru_init_state = gru_cell.zero_state(zero_state_length, dtype=tf.float32)
            # run GRU backward
            outputs, state = tf.nn.dynamic_rnn(gru_cell, inputs=input_x, initial_state=gru_init_state)
            outputs = tf.reverse_v2(outputs, [1])
        return outputs, state

    def word_attention(self, hidden_state):
        """
        this function is able to get word attention from sentence
        :param hidden_state:shape[batch_size*num_sentence,sequence_length,hidden_size*2]
        :return:hidden_state by add attention weight
        """
        """
            hidden_state_:shape [batch_size*num_sentence*sequence_length, hidden_size*2]
        """
        hidden_state_ = tf.reshape(hidden_state, shape=[-1, self.hidden_size * 2])
        """
            hidden_representation:shape [batch_size*num_sentence*sequence_length, hidden_size*2]
        """
        hidden_representation = tf.nn.tanh(tf.matmul(hidden_state_, self.W_w_attention_word) + self.W_b_attention_word)
        """
            hidden_representation:shape [batch_size*num_sentence, sequence_length, hidden_size*2]
        """
        hidden_representation = tf.reshape(hidden_representation, [-1, self.sequence_length, self.hidden_size * 2])
        """
            hidden_state_context_similiarity:shape [batch_size*num_sentence, sequence_length, hidden_size*2]
        """
        hidden_state_context_similiarity = tf.multiply(hidden_representation, self.context_vecotor_word)
        """
            attention_logits:shape [batch_size*num_sentence, sequence_length]
        """
        attention_logits = tf.reduce_sum(hidden_state_context_similiarity,
                                         axis=2)  # calculate every word sequence embedding sum
        """
            attention_logits_max:shape [batch_size*num_sentence, 1]
        """
        attention_logits_max = tf.reduce_max(attention_logits, axis=1,
                                             keep_dims=True)  # get a sentence max embedding of word
        """
             p_attention:shape [batch_size*num_sentence, sequence_length]
        """
        p_attention = tf.nn.softmax(attention_logits - attention_logits_max)
        """
             expand dimension
             p_attention_expanded:shape [batch_size*num_sentence, sequence_length, 1]
        """
        p_attention_expanded = tf.expand_dims(p_attention, axis=2)
        """
            add probability to hidden_state, shape:[batch_size*num_sentences,sequence_length,hidden_size*2]
        """
        sentence_representation = tf.multiply(p_attention_expanded,
                                              hidden_state)
        """
            shape:[batch_size*num_sentences,hidden_size*2]
        """
        sentence_representation = tf.reduce_sum(sentence_representation, axis=1)

        return sentence_representation

    def sentence_attention(self, hidden_state):
        """
        this function is able to get sentence attention from document
        :param hidden_state: shape:[batch_size, num_sentence, hidden_size*4]
        :return: [batch_size, hidden_size*4]
        """
        """
            shape:[batch_size*num_sentence, hidden_size*4]
        """
        hidden_state_ = tf.reshape(hidden_state, [-1, self.hidden_size * 4])
        """
            shape:[batch_size*num_sentence, hidden_size*2]
        """
        hidden_representation = tf.nn.tanh(tf.matmul(hidden_state_,
                                                     self.W_w_attention_sentence) + self.W_b_attention_sentence)
        """
            shape:[batch_size, num_sentence, hidden_size * 2]
        """
        hidden_representation = tf.reshape(hidden_representation, shape=[-1, self.config.num_sentence,
                                                                         self.hidden_size * 2])
        """
        attention process:
            1.get logits for each sentence in the doc.
            2.get possibility distribution for each sentence in the doc.
            3.get weighted sum for the sentences as doc representation
        """
        """
            1) get logits for each word in the sentence.
            shape:[batch_size, num_sentence, hidden_size * 2]
        """
        hidden_state_context_similiarity = tf.multiply(hidden_representation, self.context_vecotor_sentence)
        """
            that is get logit for each num_sentence.
            shape:[batch_size, num_sentence]
        """
        attention_logits = tf.reduce_sum(hidden_state_context_similiarity, axis=2)
        """
            subtract max for numerical stability (softmax is shift invariant).
            tf.reduce_max:computes the maximum of elements across dimensions of a tensor
            shape: [batch_size, 1]
        """
        attention_logits_max = tf.reduce_max(attention_logits, axis=1, keep_dims=True)
        """
            2) get possibility distribution for each word in the sentence.
            shape: [batch_size, num_sentence]
            calculate every sentence contribution degree
        """
        p_attention = tf.nn.softmax(attention_logits - attention_logits_max)
        """
           # 3) get weighted hidden state by attention vector(sentence level)
           shape: [batch_size, num_sentence, 1] 
        """
        self.p_attention_expanded = tf.expand_dims(p_attention, axis=2)
        """
            multiply all representation
            shape:[batch_size, num_sentence, hidden_size*4]
        """
        sentence_representation = tf.multiply(self.p_attention_expanded, hidden_state)
        """
            get sum
            shape:[batch_size, hidden_size*4]
        """
        sentence_representation = tf.reduce_sum(sentence_representation, axis=1)

        return sentence_representation

    def classficatoin_text_logits(self, hidden_state):
        """
        :param hidden_state: HAN hidden output
        :return:classfication result
        """
        with tf.name_scope('softmax'):
            logits = tf.nn.softmax(tf.matmul(hidden_state, self.W_softmax) +
                                   self.B_softmax)  # shape:[None,class_num]

        return logits

    def classficatoin_text_loss(self, logits):
        """
        :param logits: softmax result
        :return: loss
        """
        with tf.name_scope('loss'):
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=self.input_y)
        return loss

    def classficatoin_text_train(self, loss):
        """
        :param loss: loss
        :return: optimize
        """
        with tf.name_scope('train'):
            optim = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        return optim

    def classficatoin_text_accuarcy(self, logits):
        """
        :param logits: logits
        :return: accuracy
        """
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(self.input_y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return accuracy

    def create_gru_unit(self, hidden_size):
        """
        create gru unit
        :param hidden_size: GRU output hidden_size
        :return: GRU cell
        """
        with tf.name_scope('create_gru_cell'):
            gru_cell = rnn.GRUCell(hidden_size)
            gru_cell = rnn.DropoutWrapper(cell=gru_cell, input_keep_prob=1.0,
                                          output_keep_prob=self.gru_output_keep_prob)

        return gru_cell

    def init_weight(self):
        """
        init weights
        :return
        """
        with tf.name_scope('attention_variable'):
            self.W_w_attention_word = tf.get_variable('W_w_attention_word',
                                                      shape=[self.hidden_size * 2, self.hidden_size * 2],
                                                      initializer=self.initializer)
            self.W_b_attention_word = tf.get_variable('W_b_attention_word',
                                                      shape=[self.hidden_size * 2],
                                                      initializer=self.initializer)
            self.context_vecotor_word = tf.get_variable("what_is_the_informative_word", shape=[self.hidden_size * 2],
                                                        initializer=self.initializer)

            self.W_w_attention_sentence = tf.get_variable('W_w_attention_sentence',
                                                          shape=[self.hidden_size * 4, self.hidden_size * 2],
                                                          initializer=self.initializer)
            self.W_b_attention_sentence = tf.get_variable('W_b_attention_sentence', shape=[self.hidden_size * 2])
            self.context_vecotor_sentence = tf.get_variable('what_is_the_informative_sentence',
                                                            shape=[self.hidden_size * 2], initializer=self.initializer)

        with tf.name_scope('softmax_variable'):
            self.W_softmax = tf.get_variable('W_softmax', shape=[self.hidden_size * 4, self.config.class_num],
                                             initializer=self.initializer)
            self.B_softmax = tf.get_variable('B_softmax', shape=[self.config.class_num])