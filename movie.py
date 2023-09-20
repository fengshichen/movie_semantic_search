import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.layers import LSTM
import numpy as np

# 从文件读取数据集
with open('datasets/movies.csv', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 提取电影基础信息
movies = [line.strip().split(',') for line in lines]
movie_texts = [movie[0] + " " + movie[1] + " " + movie[2] + " " + movie[3] + " "
               + movie[4] + " " + movie[5] + " " + movie[6] + " " + movie[7] + " "
               + movie[8] + " " + movie[9] + " " + movie[10] + " " + movie[11] + " "
               + movie[12] + " " + movie[13] + " " + movie[14] + " " + movie[15] + " "
               + movie[16] + " " + movie[17] + " " + movie[18] + " " + movie[19] + " " + movie[20]
               for movie in movies]

# 数据预处理
# 词汇表的大小
vocab_size = 128322
# 词嵌入维度
embedding_dim = 300
# 每个文本序列的最大单词数。
max_length = 10000
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(movie_texts)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(movie_texts)
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)


# 支持中文的词嵌入
def load_embeddings(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            if word in word_index:
                idx = word_index[word]
                embedding_matrix[idx] = np.array(values[1:], dtype=np.float32)
    return embedding_matrix


embedding_matrix = load_embeddings('sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5', word_index,
                                         embedding_dim)

# 构建模型
model = Sequential([
    # Embedding(vocab_size, embedding_dim, input_length=max_length),
    Embedding(vocab_size, embedding_dim, input_length=max_length, weights=[embedding_matrix], trainable=False),
    # LSTM模型结构
    LSTM(64, return_sequences=True),
    LSTM(64),
    Dense(64, activation='relu'),
    Dense(max_length, activation='relu')
])

model.compile(loss='mse', optimizer='adam')

model.fit(padded, padded, epochs=10)

model.save('result_models/semantic_search_model')
