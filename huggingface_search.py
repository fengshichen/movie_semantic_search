import numpy as np
from transformers import BertTokenizer, TFBertModel

with open('./datasets/movies_full.csv', 'r', encoding='utf-8') as f:
    lines = f.readlines()

movies = [line.strip().split(',') for line in lines]

# 加载模型和分词器
model_path = './transformers_model'
tokenizer_path = './transformers_model'
model = TFBertModel.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

# 加载嵌入
embeddings_path = './transformers_model/movie_embeddings.npy'
movie_embeddings = np.load(embeddings_path)


def semantic_search(query, model, tokenizer, movie_embeddings, movies, top_n):
    query_input = tokenizer(query, return_tensors="tf", padding=True, truncation=True, max_length=128)
    query_embedding = model(query_input['input_ids']).last_hidden_state[:, 0, :].numpy()

    cos_similarities = np.dot(movie_embeddings, query_embedding.T).squeeze()
    most_similar_indices = np.argsort(cos_similarities)[-top_n:][::-1]

    return [movies[i] for i in most_similar_indices]


query = "推荐科幻影片，喜剧，让人震撼"
print(semantic_search(query, model, tokenizer, movie_embeddings, movies, top_n=10))
