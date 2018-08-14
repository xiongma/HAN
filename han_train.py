import numpy as np

import tensorflow as tf
import tensorflow.contrib.keras as kr

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from keras.utils import to_categorical

from news_w2v.news_vec import NewsW2V
from data.get_data import DataProcess
from tagjieba.instance import TagJieba

from han_config import HierarchicalAttentionConfig
from han_model import HierarchicalAttention

class HierarchicalAttentionTrain(object):
    def __init__(self):
        self.han_config = HierarchicalAttentionConfig()

        self.news_word2Vec_model = NewsW2V()
        self.word2vec_vocal_dict = dict(zip(self.news_word2Vec_model.w2v_model.wv.index2word,
                                   range(len(self.news_word2Vec_model.w2v_model.wv.index2word))))
        print('init word2vec success....')

        self.tag_jieba = TagJieba()
        print('init tag jieba success....')

        self.dp = DataProcess()
        self.han_model = HierarchicalAttention(config=self.han_config,
                                               embedding=self.news_word2Vec_model.w2v_model.wv.vectors)

    def word_to_id(self, words):
        """
        this function is able to get word id from word2vec vocals
        :param words: words
        :return: words id
        """
        words_id = []
        for word in words:
            try:
                words_id.append(self.word2vec_vocal_dict[word])
            except:
                pass

        return words_id

    def batch_iter(self, X, y, batch_size):
        """
        this function is able to get batch iterate of total data set
        :param X: X
        :param y: y
        :param batch_size: batch size
        :return: batch iterate
        """
        data_len = len(X)
        num_batch = int((data_len - 1) / batch_size) + 1
        for i in range(num_batch):
            start_id = i * batch_size
            end_id = min((i + 1) * batch_size, data_len)

            yield X[start_id:end_id], y[start_id:end_id]

    def train(self):
        """
        this function is able start train HAN model
        :return:
        """
        # get data
        news = self.dp.read_data(self.han_config.advertising.all_path)
        news_content = [data[1] for data in news]
        news_label = [data[2] for data in news]

        # cut content
        content_cut = [self.tag_jieba.lcut(data) for data in news_content]

        # data set to id and padding immobilization sequence length
        dataset = [self.word_to_id(data) for data in content_cut]
        X = kr.preprocessing.sequence.pad_sequences(dataset, self.han_config.sequence_length)
        y = to_categorical(news_label, num_classes=self.han_config.class_num)

        # split data set
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=1-self.han_config.train_rate)

        # train params
        steps = 0
        best_accuracy = 0
        last_improved = 0
        early_stop = False

        # model saver
        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            for epoch in range(self.han_config.epoch):
                batch_iterate = self.batch_iter(X_train, y_train, self.han_config.batch_size)

                for input_x, input_y in batch_iterate:
                    train_accuracy = session.run(self.han_model.accuracy,
                            feed_dict={self.han_model.batch_size: self.han_config.batch_size,
                                        self.han_model.learning_rate: self.han_config.learning_rate,
                                        self.han_model.input_x: input_x,
                                        self.han_model.input_y: input_y})

                    train_loss = session.run(self.han_model.loss,
                            feed_dict={self.han_model.batch_size: self.han_config.batch_size,
                                        self.han_model.learning_rate: self.han_config.learning_rate,
                                        self.han_model.input_x: input_x,
                                        self.han_model.input_y: input_y})

                    if steps % self.han_config.num_train == 0:
                        test_accuracy = session.run(self.han_model.accuracy,
                                                    feed_dict={self.han_model.batch_size: self.han_config.batch_size,
                                                            self.han_model.learning_rate: self.han_config.learning_rate,
                                                            self.han_model.input_x: X_val,
                                                            self.han_model.input_y: y_val})

                        test_loss = session.run(self.han_model.loss,
                                                    feed_dict={self.han_model.batch_size: self.han_config.batch_size,
                                                            self.han_model.learning_rate: self.han_config.learning_rate,
                                                            self.han_model.input_x: X_val,
                                                            self.han_model.input_y: y_val})

                        if test_accuracy > best_accuracy:
                            best_accuracy = test_accuracy
                            last_improved = steps
                            saver.save(session, save_path=self.han_config.model_path)

                        # early stop
                        if steps - last_improved > self.han_config.require_improved:
                            print("No optimization for a long time, auto-stopping...")
                            early_stop = True
                            break

                        print('{0} train accuracy is {1}, train loss is {2}'.format(steps, train_accuracy,
                                                                                    train_loss))
                        print('{0} test accuracy is {1}, test loss is {2}'.format(steps, test_accuracy,
                                                                                    test_loss))

                    # optimize loss
                    session.run(self.han_model.optim,
                                feed_dict={self.han_model.batch_size: self.han_config.batch_size,
                                           self.han_model.learning_rate: self.han_config.learning_rate,
                                           self.han_model.input_x: input_x,
                                           self.han_model.input_y: input_y})

                    steps = steps + 1

                    if early_stop:
                        logits = session.run(self.han_model.logits,
                                                    feed_dict={self.han_model.batch_size: self.han_config.batch_size,
                                                               self.han_model.learning_rate: self.han_config.learning_rate,
                                                               self.han_model.input_x: X_val,
                                                               self.han_model.input_y: y_val})

                        print(' f1 is {0}'.format(classification_report(np.argmax(y_val, axis=1),
                                                                        np.argmax(logits, axis=1))))
                        return

                    if epoch == self.han_config.epoch - 2:
                        logits = session.run(self.han_model.logits,
                                              feed_dict={self.han_model.batch_size: self.han_config.batch_size,
                                                         self.han_model.learning_rate: self.han_config.learning_rate,
                                                         self.han_model.input_x: X_val,
                                                         self.han_model.input_y: y_val})

                        print(' f1 is {0}'.format(classification_report(np.argmax(y_val, axis=1),
                                                                        np.argmax(logits, axis=1))))

train = HierarchicalAttentionTrain()
train.train()