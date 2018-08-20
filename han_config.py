
import data

class HierarchicalAttentionConfig:
    def __init__(self):
        self.learning_rate = 0.001
        self.num_sentence = 10 # num sentence in documents
        self.sequence_length = 200 # sequence length of each sentence
        self.hidden_size = 300
        self.advertising = data.Advertising()
        self.gru_output_keep_prob = 0.5
        self.class_num = 2
        self.batch_size = 160
        self.epoch = 10 # train epoch
        self.train_rate = 0.8 # this is train data set rate of total data set
        self.num_train = 100 # how many number display train accuracy
        self.model_path = '/model'
        self.require_improved = 500