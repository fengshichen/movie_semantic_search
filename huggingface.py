import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel

#HuggingFace
# 数据集
with open('./datasets/movies_full.csv', 'r', encoding='utf-8') as f:
    lines = f.readlines()

movies = [line.strip().split(',') for line in lines]
movie_texts = [" ".join(movie) for movie in movies]

vocab_file = 'bert/vocab.txt'
tokenizer = BertTokenizer(vocab_file)
inputs = tokenizer(movie_texts, return_tensors="tf", padding=True, truncation=True, max_length=128)

model = TFBertModel.from_pretrained("bert/bert-base-chinese/")

batch_size = 500
all_movie_embeddings = []

for i in range(0, len(movie_texts), batch_size):
    batch_texts = movie_texts[i:i+batch_size]
    inputs = tokenizer(batch_texts, return_tensors="tf", padding=True, truncation=True, max_length=128)
    batch_output = model(inputs['input_ids'])
    batch_embeddings = batch_output.last_hidden_state[:, 0, :]
    all_movie_embeddings.append(batch_embeddings)

movie_embeddings = tf.concat(all_movie_embeddings, axis=0)
movie_embeddings = movie_embeddings.numpy()
np.save('./transformers_model/movie_embeddings.npy', movie_embeddings)

model.save_pretrained('./transformers_model')
tokenizer.save_pretrained('./transformers_model')
