# HAN
Hierarchical Attention Networks for Document Classification

1.this model need data set is excel file, which has 4 columns
  a.first column is news id
  b.second column is news title
  c.third column is news content
  d.fourth column is news label
2.this model need word2vec, add this model folder under HAN folder

Run train
  1.modify han_config.py
    a.modify gru output hidden size
    b.modify data_set_path
    c.modify class numbers
  2. run han_train.py

Run prediction
  1.run han_prediction.py function