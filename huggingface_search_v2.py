import numpy as np
from transformers import BertTokenizer, TFBertModel
import jieba

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

# 从数据源中到出所有电影分类
category_to_movie_index = {}
for idx, movie in enumerate(movies):
    categories = movie[8].split("/")
    for category in categories:
        if category not in category_to_movie_index:
            category_to_movie_index[category] = []
        category_to_movie_index[category].append(idx)


def semantic_search(query, model, tokenizer, movie_embeddings, movies, top_n):
    query_input = tokenizer(query, return_tensors="tf", padding=True, truncation=True, max_length=128)
    query_embedding = model(query_input['input_ids']).last_hidden_state[:, 0, :].numpy()

    cos_similarities = np.dot(movie_embeddings, query_embedding.T).squeeze()

    # 使用分词器将用户的query进行分词
    query_terms = jieba.lcut(query)
    # 判断query中是否命中电影分类，若命中则将权重提高
    for possible_category in category_to_movie_index.keys():
        for term in query_terms:
            if term in possible_category:
                for idx in category_to_movie_index[possible_category]:
                    cos_similarities[idx] += 10.0

    most_similar_indices = np.argsort(cos_similarities)[-top_n:][::-1]

    return [movies[i] for i in most_similar_indices]


query = "给我推荐几个爱情电影"
print(semantic_search(query, model, tokenizer, movie_embeddings, movies, top_n=10))
