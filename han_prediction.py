
import numpy as np
import tensorflow as tf
import tensorflow.contrib.keras as kr

from news_w2v.news_vec import NewsW2V
from tagjieba.instance import TagJieba

from han_model import HierarchicalAttention
from han_config import HierarchicalAttentionConfig

class HierarchicalAttentionPrediction(object):
    def __init__(self):
        self.han_config = HierarchicalAttentionConfig()

        self.news_word2Vec_model = NewsW2V()
        self.word2vec_vocal_dict = dict(zip(self.news_word2Vec_model.w2v_model.wv.index2word,
                                            range(len(self.news_word2Vec_model.w2v_model.wv.index2word))))
        print('init word2vec success....')

        self.tag_jieba = TagJieba()
        print('init tag jieba success....')

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

    def news_classification(self, title, content):
        """
        this function is able to classification news
        :param title: news title
        :param content: news content
        :return: news label
        """
        title_words = self.tag_jieba.cut(title)
        content_words = self.tag_jieba.cut(content)
        words = title_words + content_words
        words_id = self.word_to_id(words)
        words_id = np.reshape(words_id, [-1, len(words_id)])
        words_id = kr.preprocessing.sequence.pad_sequences(words_id, self.han_config.sentence_length)

        # restore model
        session = tf.Session()
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(session, self.han_config.model_path)

        # prediction
        logits = session.run(self.han_model.logits,
                    feed_dict={
                        self.han_model.learning_rate: self.han_config.learning_rate,
                        self.han_model.input_x: words_id
                    })

        return logits

if __name__ == "__main__":
    prediction = HierarchicalAttentionPrediction()
    logits = prediction.news_classification(title="""卧室""",
                                   content="""他是""")

    print(logits)
