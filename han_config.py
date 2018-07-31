
import data

class HierarchicalAttentionConfig:
    def __init__(self):
        self.learning_rate = 0.001
        self.num_sentence = 10 # num sentence in documents
        self.sequence_length = 200 # sequence length of each sentence
        self.hidden_size = 300
        self.education = data.Education()
        self.gru_output_keep_prob = 0.5
        self.class_num = 21
        self.batch_size = 32