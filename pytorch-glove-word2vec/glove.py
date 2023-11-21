import torch, pickle, os
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

import numpy as np
from tqdm import tqdm

from tools import CorpusPreprocess, VectorEvaluation

# params
x_max = 100                     #计算权重的参数
alpha = 0.75                    #计算权重的参数
epoches = 1                     #训练轮数
min_count = 0                   #用于过滤词频太低的词语
batch_size = 512                #训练过程中的批次大小，表示每次更新模型参数时所使用的样本数。
windows_size = 5                #GloVe模型中的上下文窗口大小，表示在训练中考虑的词语上下文的范围。
vector_size = 300               #词向量的维度，表示每个词语在训练后将被表示为一个具有多少维度的向量。
learning_rate = 0.001           #训练时使用的学习率，表示在模型参数更新时每次迭代的步长大小。
glove_path = 'glmodel/'         #存储训练好的GloVe模型的路径
if not os.path.exists(glove_path): os.makedirs(glove_path) #如果路径不存在，则会创建。
model_name = 'glove_{}.pkl'.format(str(vector_size))        #存储训练好的GloVe模型文件的名称，包括词向量的维度。
glove_model_file = os.path.join(glove_path, model_name)     #最终的GloVe模型文件的完整路径，由glove_path和model_name拼接而成。
# print('模型保存路径：',glove_model_file)
# get gpu
use_gpu = torch.cuda.is_available()                     #检测是否支持GPU如果支持则为True否则非False


# calculation weight
def fw(X_c_s):
    return (X_c_s / x_max) ** alpha if X_c_s < x_max else 1         #对于权重值的调整


class Glove(nn.Module):
    def __init__(self, vocab_size, vector_size):
        super(Glove, self).__init__()
        # center words weight and biase
        self.c_weight = nn.Embedding(len(vocab_size), vector_size,      #用于表示中心词的权重
                                     _weight=torch.randn(len(vocab_size),
                                                         vector_size,
                                                         dtype=torch.float,
                                                         requires_grad=True) / 100)

        self.c_biase = nn.Embedding(len(vocab_size), 1, _weight=torch.randn(len(vocab_size),        #用于表示中心词的偏置
                                                                            1, dtype=torch.float,
                                                                            requires_grad=True) / 100)

        # surround words weight and biase
        self.s_weight = nn.Embedding(len(vocab_size), vector_size,                  #表示周围词的偏置和权重
                                     _weight=torch.randn(len(vocab_size),
                                                         vector_size, dtype=torch.float,
                                                         requires_grad=True) / 100)

        self.s_biase = nn.Embedding(len(vocab_size), 1,
                                    _weight=torch.randn(len(vocab_size),
                                                        1, dtype=torch.float,
                                                        requires_grad=True) / 100)

    def forward(self, c, s):        #定义了 Glove 类中的前向传播方法 forward。
        c_w = self.c_weight(c)      #获取中心词的权重和偏置
        c_b = self.c_biase(c)       #获取中心词的权重和偏置
        s_w = self.s_weight(s)      #获取周围词的权重和偏置
        s_b = self.s_biase(s)       #获取周围词的权重和偏置
        return torch.sum(c_w.mul(s_w), 1, keepdim=True) + c_b + s_b     #计算元素积并求和：torch.sum(c_w.mul(s_w), 1, keepdim=True)再加上权重和偏置


# read data
class TrainData(Dataset):           #该类用于在 PyTorch 中加载和处理数据。
    def __init__(self, coo_matrix):     #稀疏矩阵 coo_matrix 作为输入。
        self.coo_matrix = [((i, j), coo_matrix.data[i][pos]) for i, row in enumerate(coo_matrix.rows) for pos, j in
                           enumerate(row)]      #将稀疏矩阵的非零元素转换为一个列表 self.coo_matrix，其中每个元素是一个元组，包含中心词索引、周围词索引和关系值。

    def __len__(self):          #这个方法返回数据集的长度，即数据集中样本的数量。在这里，返回 self.coo_matrix 的长度。
        return len(self.coo_matrix)

    def __getitem__(self, idex):        #这个方法用于获取数据集中指定索引 index 处的样本。
        sample_data = self.coo_matrix[idex]
        sample = {"c": sample_data[0][0],
                  "s": sample_data[0][1],
                  "X_c_s": sample_data[1],
                  "W_c_s": fw(sample_data[1])}
        return sample


def loss_func(X_c_s_hat, X_c_s, W_c_s):     #损失函数
    X_c_s = X_c_s.view(-1, 1)               #将其改为列向量
    W_c_s = X_c_s.view(-1, 1)               #将其改为列向量
    loss = torch.sum(W_c_s.mul((X_c_s_hat - torch.log(X_c_s)) ** 2))#计算损失，具体的公式为：在我csdn上GLOVE损失函数中
    return loss


# save vector
def save_word_vector(file_name, corpus_preprocessor, glove):
    with open(file_name, "w", encoding="utf-8") as f:       #打开一个文件以写入词向量。file_name 是保存词向量的文件名。
        if use_gpu:                                         #是否支持GPU
            c_vector = glove.c_weight.weight.data.cpu().numpy() #载入中心词和周围词的权重
            s_vector = glove.s_weight.weight.data.cpu().numpy()
            vector = c_vector + s_vector                        #相加
        else:
            c_vector = glove.c_weight.weight.data.numpy()       #在CPU上运行
            s_vector = glove.s_weight.weight.data.numpy()
            vector = c_vector + s_vector
        # try:
        #     with open('output/vector.pkl', 'wb') as p:
        #         pickle.dump(vector, p)
        #     print('vector的shape', vector.shape)
        # except:
        #     print('打印vector的shape有误')
        for i in tqdm(range(len(vector))):                  #使用 tqdm 库中的 tqdm(range(len(vector))) 来显示遍历的进度。
            word = corpus_preprocessor.idex2word[i]         #使用一个循环遍历词向量，对每个词根据其索引获取对应的词，并将词向量以词与其向量表示的形式写入文件中。
            s_vec = vector[i]
            s_vec = [str(s) for s in s_vec.tolist()]
            write_line = word + " " + " ".join(s_vec) + "\n"
            f.write(write_line)
        print("词向量保存完成！好耶\(^o^)/~")


def train_model(epoches, corpus_file_name):
    corpus_preprocessor = CorpusPreprocess(corpus_file_name, min_count)         #创建 CorpusPreprocess 对象，用于对语料进行预处理，包括构建共现矩阵（co-occurrence matrix）等。
    coo_matrix = corpus_preprocessor.get_cooccurrence_matrix(windows_size)
    vocab = corpus_preprocessor.get_vocab()         #根据共现矩阵构建词汇表，并创建 GloVe 模型对象 glove。
    glove = Glove(vocab, vector_size)

    print(glove)
    if os.path.isfile(glove_model_file):            #如果指定的 GloVe 模型文件存在，载入模型参数。
        glove.load_state_dict(torch.load(glove_model_file))
        print('载入模型{}'.format(glove_model_file))
    if use_gpu:                                     #如果使用 GPU，将模型移到 GPU 上。
        glove.cuda()
    optimizer = torch.optim.Adam(glove.parameters(), lr=learning_rate)  #创建 Adam 优化器，用于更新模型参数。

    train_data = TrainData(coo_matrix)      #创建 TrainData 对象，将共现矩阵转换为适用于模型训练的数据集。
    data_loader = DataLoader(train_data,    #使用 DataLoader 加载数据，设置 batch_size、shuffle 等参数。
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=2,
                             pin_memory=True)

    steps = 0
    for epoch in range(epoches):            #开始训练循环，外层循环是每个 epoch，内层循环是每个 batch。
        print(f"currently epoch is {epoch + 1}, all epoch is {epoches}")
        avg_epoch_loss = 0
        for i, batch_data in enumerate(data_loader):
            c = batch_data['c']
            s = batch_data['s']
            X_c_s = batch_data['X_c_s']
            W_c_s = batch_data["W_c_s"]

            if use_gpu:                 #将数据移动到 GPU（如果使用 GPU）。
                c = c.cuda()
                s = s.cuda()
                X_c_s = X_c_s.cuda()
                W_c_s = W_c_s.cuda()

            W_c_s_hat = glove(c, s)     #使用定义的GloVe模型 glove 进行前向传播，得到预测的词向量。
            loss = loss_func(W_c_s_hat, X_c_s, W_c_s)   #计算预测的词向量与目标词向量之间的损失，使用的损失函数是之前定义的 loss_func。
            optimizer.zero_grad()           #在每个batch开始时，清零之前的梯度，以防止梯度累积。
            loss.backward()                 #反向传播，计算梯度。
            optimizer.step()                #根据梯度更新模型参数。
            avg_epoch_loss += loss / len(train_data)    #将当前batch的损失累加到总的epoch损失中，以后计算平均损失。
            if steps % 1000 == 0:                       #每1000个训练步骤打印一次损失。
                print(f"Steps {steps}, loss is {loss.item()}")      #打印当前训练步骤和损失值。
            steps += 1      #更新训练步骤的计数器。
        print(f"Epoches {epoch + 1}, complete!, avg loss {avg_epoch_loss}.\n")      #循环结束后，计算并打印平均损失，表示当前epoch的训练完成。
    save_word_vector(save_vector_file_name, corpus_preprocessor, glove)                 #保存训练得到的词向量到指定文件中。
    torch.save(glove.state_dict(), glove_model_file)                            #保存训练得到的模型参数。

def get_dict_and_embedding(model_dir,embed_dir,vocabulary_dir,embed_dim):
    word2id={'<pad>':0}     #初始化一个字典，将 <pad> 这个特殊词添加到字典中，并设置其索引为0。
    index=1                 #初始化一个索引计数器，用于给词汇表中的其他词分配索引。
    embeddings=[]           #初始化一个列表，用于存储词向量。
    embeddings.append([0]*embed_dim)            #在词向量列表中添加一个全为0的向量，用于表示 <pad>。
    with open (model_dir,'r', encoding='utf-8') as f:   #打开存储训练好的 GloVe 模型的文件 model_dir。
        lines=f.readlines()
        for line in lines:
            if len(line.strip().split())<3:
                continue
            word=line.strip().split()[0]
            data=line.strip().split()[1:]
            word2id[word]=index              #遍历文件中的每一行，解析每个词及其对应的词向量，将词添加到 word2id 字典中，
            index+=1
            try:
                assert len(data)==embed_dim
            except:
                print(len(data),embed_dim)
            embeddings.append(data)         #将词向量添加到 embeddings 列表中。
    embeddings=np.array(embeddings)         #将 embeddings 列表转换为 NumPy 数组，形状为 (词汇量大小+1, embed_dim)。
    print('glove embeddings的‘形状’为：({}*{})=={}'.format(index,embed_dim,embeddings.shape))       #打印词向量的形状信息。
    with open(embed_dir,'wb') as p:     #将词向量保存为二进制文件（embed_dir），以备后续加载使用。
        pickle.dump(embeddings,p)
    with open(vocabulary_dir,'wb') as p1:       #将词典保存为二进制文件（vocabulary_dir），以备后续加载使用。
        pickle.dump(word2id,p1)
    print('完成！')            #打印提示信息，表示保存过程完成。

if __name__ == "__main__":
    # file_path
    if not os.path.exists('output/'): os.makedirs('output/')        #创建输出目录：检查是否存在名为 'output/' 的目录，如果不存在则创建。
    save_vector_file_name = "output/glove.txt"                      #定义保存词向量、图片和嵌入矩阵的文件路径：  #保存GloVe训练得到的词向量的文件路径。
    save_picture_file_name = "output/glove.png"                     #保存生成的相似性图的文件路径。
    embed_dir='output/glove_{}.pkl'.format(str(vector_size))        #保存嵌入矩阵的文件路径。
    vocabulary_dir='output/vocabulary.pkl'                          #保存词典的文件路径。
#   corpus_file_name = 'data/train_corpus/MOSItext.txt'             #定义语料文件路径 corpus_file_name，这里使用了一个训练语料库的文件路径。
    corpus_file_name = 'data/MOSItext.txt'
    train_model(epoches, corpus_file_name)                          #调用 train_model 函数进行模型训练，使用指定的训练轮数和语料文件。
    vec_eval = VectorEvaluation(save_vector_file_name)              #创建 VectorEvaluation 对象 vec_eval，用于对训练得到的词向量进行评估。
    vec_eval.drawing_and_save_picture(save_picture_file_name)     #注释掉了生成相似性图并保存图片的代码，因为这部分被注释掉了。
    #vec_eval.get_similar_words("加拿大")                             #调用 get_similar_words 函数，通过给定的词来获取相似的词语。
    # vec_eval.get_similar_words("男人")
    get_dict_and_embedding(model_dir=save_vector_file_name,embed_dir=embed_dir,vocabulary_dir=vocabulary_dir,embed_dim=vector_size)
    #调用 get_dict_and_embedding 函数，将训练得到的词向量保存为嵌入矩阵和词典。
