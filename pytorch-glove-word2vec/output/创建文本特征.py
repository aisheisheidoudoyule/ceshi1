import torch
import pickle

# 步骤1：加载模型和词典
glove_model_file = 'glove_300.pkl'
vocabulary_file = 'vocabulary.pkl'

# 使用 torch.load 加载模型
glove_model = torch.load(glove_model_file, map_location=torch.device('cpu'))


with open(vocabulary_file, 'rb') as f:
    vocabulary = pickle.load(f)

# 步骤2：将文本转换为词索引
def text_to_indices(text, vocabulary):
    return [vocabulary.get(word, 0) for word in text.split()]

# 步骤3：获取词向量
def get_word_vectors(indices, glove_model):
    # 直接使用 nn.Embedding 的 weight 属性获取词向量
    weights = glove_model['c_weight.weight']
    return weights[indices]

# 步骤4：特征表示
def text_to_features(text, vocabulary, glove_model):
    indices = text_to_indices(text, vocabulary)
    vectors = get_word_vectors(torch.tensor(indices), glove_model)
    # 这里可以根据需要对词向量进行合适的操作，例如求平均值
    features = torch.mean(vectors, dim=0)
    return features

# 示例用法
text_data = "your input text here"
features = text_to_features(text_data, vocabulary, glove_model)
print(features)
