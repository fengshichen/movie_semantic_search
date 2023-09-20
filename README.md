# movie_semantic_search
### 一、项目背景：
基于影片库的语义搜索

### 二、技术背景
结合python3+tensorflow，通过给定电影资料数据集进行模型训练，基于模型进行语义搜索

### 三、环境搭建
参考官方文档安装python3+tensorflow环境，参考资料：https://www.tensorflow.org/install  
由于基于cpu的模型训练耗时长，因此将环境迁移至pc上借助显卡进行gpu运算，参考资料：https://www.cnblogs.com/LandWind/p/win11-cuda-cudnn-Tensorflow-GPU-env-start.html  

### 四、V1.0版本
movie.py：读取豆瓣电影数据集文件datasets/movies.csv进行模型训练，并保存模型文件至./result_models  
movie_search.py：基于训练模型进行语义搜索  
初始使用的基本模型进行训练，但是搜索结果不理想，原因可能有：  
1.模型结构，针对复杂的语义搜索可能需要更复杂的模型结构，例如RNN、LSTM、GRU或Transformer  
2.数据问题，针对中文可能需要预处理分词  
3.相似度计算，使用余弦相似度而不是点积  
4.训练样本不足  
因此做以下改进优化：  
1.使用预训练好的中文模型嵌入，load_glove_embeddings方法  
2.语义搜索计算相似度的时候使用余弦相似度，cosine_similarity方法  

### 五、V2.0版本
经过查阅资料。huggingface工具拥有优秀的支持中文的模型和分词器，因此进行优化
huggingface.py 导入BERT模型（bert-base-chinese），Bert分词器，基于给定的数据集训练模型，保存模型、分词器、嵌入模型
huggingface_search.py 读取上述保存的模型，做语义搜索

### 六、V3.0版本
经过测试，2.0版本程序推荐结果较1.0版本有很大改善  
继续思考用户搜索词，针对电影场景，用户搜索词中大概率会出现电影分类（喜剧、动作...），电影语言（英语、西班牙语、中文...）  
例如：给我推荐几个动作英文电影  
因此针对搜索代码进行改造，支持将命中的关键词权重提高，代码中增加了对电影分类命中权重的提高，huggingface_search_v2.py


