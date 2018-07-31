import tensorflow as tf
import tensorflow.contrib.keras as kr

from news_w2v.news_vec import NewsW2V
from data.getdata import TagJieba, DataProcess

from han_config import HierarchicalAttentionConfig
from han_model import HierarchicalAttention

class HAN_Train(object):
    def __init__(self):
        self.han_config = HierarchicalAttentionConfig()

        self.news_word2Vec_model = NewsW2V()
        self.word2vec_vocal_dict = dict(zip(self.news_word2Vec_model.w2v_model.wv.index2word,
                                   range(len(self.news_word2Vec_model.w2v_model.wv.index2word))))
        print('init word2vec success....')
        self.tag_jieba = TagJieba()
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
        # return [self.word2vec_vocal_dict[word] for word in words]

    def train(self):
        """
        this function is able train HAN model
        :return:
        """
        news = self.dp.read_data(self.han_config.education.garbage_path)
        news_content = [data[0] for data in news]
        content_cut = [self.tag_jieba.lcut(data) for data in news_content]
        dataset = [self.word_to_id(data) for data in content_cut]
        X = kr.preprocessing.sequence.pad_sequences(dataset,
                                                    int(self.han_config.sequence_length / self.han_config.num_sentence))

        session = tf.Session()
        session.run(tf.initialize_local_variables())
        logits = session.run(self.han_model.p_attention_expanded,
                             feed_dict={self.han_model.batch_size: self.han_config.batch_size,
                                        self.han_model.learning_rate: self.han_config.learning_rate,
                                        self.han_model.input_x: X})
        print(logits)

train = HAN_Train()
train.train()