import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.layers import LSTM
import numpy as np

with open('datasets/movies.csv', 'r', encoding='utf-8') as f:
    lines = f.readlines()

movies = [line.strip().split(',') for line in lines]
movie_texts = [movie[0] + " " + movie[1] + " " + movie[2] + " " + movie[3] + " "
               + movie[4] + " " + movie[5] + " " + movie[6] + " " + movie[7] + " "
               + movie[8] + " " + movie[9] + " " + movie[10] + " " + movie[11] + " "
               + movie[12] + " " + movie[13] + " " + movie[14] + " " + movie[15] + " "
               + movie[16] + " " + movie[17] + " " + movie[18] + " " + movie[19] + " " + movie[20]
               for movie in movies]

vocab_size = 128322
embedding_dim = 300
max_length = 10000
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(movie_texts)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(movie_texts)
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)


def load_glove_embeddings(filepath, word_index, embedding_dim):
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


embedding_matrix = load_glove_embeddings('sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5', word_index,
                                         embedding_dim)

model = Sequential([
    # Embedding(vocab_size, embedding_dim, input_length=max_length),
    Embedding(vocab_size, embedding_dim, input_length=max_length, weights=[embedding_matrix], trainable=False),
    LSTM(64, return_sequences=True),
    LSTM(64),
    Dense(64, activation='relu'),
    Dense(max_length, activation='relu')
])

loaded_model = load_model('result_models/semantic_search_model')


def cosine_similarity(A, B):
    dot_product = np.dot(A, B)
    norm_a = np.linalg.norm(A)
    norm_b = np.linalg.norm(B)
    return dot_product / (norm_a * norm_b)


def semantic_search(query, top_n=3):
    query_sequence = tokenizer.texts_to_sequences([query])
    query_padded = pad_sequences(query_sequence, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    query_embedding = loaded_model.predict(query_padded)

    movie_embeddings = loaded_model.predict(padded)
    # similarities = np.dot(movie_embeddings, query_embedding.T).squeeze()
    similarities = [cosine_similarity(query_embedding[0], movie_embedding) for movie_embedding in movie_embeddings]

    # most_similar_indices = similarities.argsort()[-top_n:][::-1]
    most_similar_indices = np.argsort(similarities)[-top_n:][::-1]

    return [movies[i] for i in most_similar_indices]


query = "科幻电影"
print(semantic_search(query, top_n=10))
