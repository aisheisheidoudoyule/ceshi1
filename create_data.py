## Credits: https://github.com/SenticNet/contextual-utterance-level-multimodal-sentiment-analysis ##
## Authors: Devamanyu Hazarika, Soujanya Poria ##
## Modified for Python 3.5 ##

import numpy as np, pandas as pd
from collections import defaultdict
import pickle
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()

#设置随机种子，为了使实验可重复，默认为17
np.random.seed(17)

# 从 "./data/transcripts.csv" 文件中读取数据，header=None 表示没有列名，然后将其转换为 NumPy 数组。
#transcripts一般为访谈类的口头表达，一行为一个样本
pre_data = np.asarray(pd.read_csv("./data/transcripts.csv" , header=None))

#从 "./data/text_train.csv" 文件中读取数据，同样没有列名。      文本模态训练数据
train = pd.read_csv("./data/text_train.csv", header=None)
#从 "./data/text_test.csv" 文件中读取测试数据，同样没有列名。     文本模态
test = pd.read_csv("./data/text_test.csv", header=None)
#将训练数据转换为 NumPy 数组。
train = np.asarray(train)
#将测试数据转换为 NumPy 数组。
test = np.asarray(test)
#从训练数据中提取第一列，并将其转换为整数型数组。这里的 train[:,0] 表示取所有行的第一列数据。
train_index = np.asarray(train[:,0], dtype = 'int')
#从测试数据中提取第一列，并将其转换为整数型数组。同样，这里的 test[:,0] 表示取所有行的第一列数据。
test_index = np.asarray(test[:,0], dtype = 'int')

# print("刚生成的维度")
# print(train.shape)



def main(name):

    #路径设置，当传入text时name为。。以此类推
    path = "./data/"+name+"/"+name
    #将路径输出
    print (path)
    #defaultdict 允许指定默认值的数据类型，即在字典中访问不存在的键时，会自动为该键分配默认值。
    train_video_mapping=defaultdict(list)
    #创建列表，下面同理
    train_video_mapping_index=defaultdict(list)
    test_video_mapping=defaultdict(list)
    test_video_mapping_index=defaultdict(list)

    #这一部分使用 pandas 库的 read_csv 函数来读取CSV文件。path+"_train0.csv" 是训练数据文件的路径，header=None 意味着数据文件中没有列标题，所有数据都将被视为内容。
    data_train = np.asarray(pd.read_csv(path+"_train0.csv", header=None))
    #测试数据
    data_test = np.asarray(pd.read_csv(path+"_test0.csv", header=None))

    #这是一个循环，遍历训练数据的索引。
    for i in range(train_index.shape[0]):
        #pre_data[train_index[i]][0]：通过 train_index[i] 获取当前训练数据的标识。
        #使用 _ 进行右侧拆分，然后取得拆分后的第一个部分，这样可以获得视频标识。
        #train_video_mapping[...]：将视频标识作为键，将训练数据的索引 train_index[i] 添加到对应视频标识的值列表中。如果视频标识已经存在于字典中，就直接添加到对应的值列表；如果不存在，就创建一个新的键值对。
        train_video_mapping[pre_data[train_index[i]][0].rsplit("_",1)[0] ].append(train_index[i])
        #这一行代码的目的是将训练数据的帧索引（通过从索引中提取的帧索引）按照视频标识进行分组，类似于上述的过程。
        train_video_mapping_index[pre_data[train_index[i]][0].rsplit("_",1)[0] ].append( int(pre_data[train_index[i]][0].rsplit("_",1)[1]) )
    #同上
    for i in range(test_index.shape[0]):
        test_video_mapping[pre_data[test_index[i]][0].rsplit("_",1)[0] ].append(test_index[i])
        test_video_mapping_index[pre_data[test_index[i]][0].rsplit("_",1)[0] ].append( int(pre_data[test_index[i]][0].rsplit("_",1)[1]) )

    #创建两个字典，其中 i 是索引位置，c 是 train_index 中的元素。
    train_indices = dict((c, i) for i, c in enumerate(train_index))
    test_indices = dict((c, i) for i, c in enumerate(test_index))

    #首先，初始化 max_len 变量为 0，这是用来存储所有视频中最大的帧数。
    max_len = 0
    #遍历训练集视频字典 (train_video_mapping) 的键值对。
    for key,value in train_video_mapping.items():
        #key 是视频标识。value 是对应视频的帧索引列表。
        max_len = max(max_len , len(value))
    for key,value in test_video_mapping.items():
        max_len = max(max_len, len(value))

    #这行代码创建了一个名为 pad 的 NumPy 数组，其中的元素都是 0。这个数组用于在训练数据中的每个视频的帧数不足 max_len 时进行填充。
    #data_train[0]：获取训练数据的第一列，通常用于存储特征。
    #data_train[0][:-1]：获取特征列的所有元素，除了最后一个元素。这通常是因为数据的最后一列可能是标签。
    #data_train[0][:-1].shape[0]：获取特征列的长度，即特征的数量。
    #[0 for i in range(data_train[0][:-1].shape[0])]：使用列表推导创建一个包含 0 的列表，列表的长度与特征数量相同。
    pad = np.asarray([0 for i in range(data_train[0][:-1].shape[0])])

    print ("Mapping train")

    #创建列表
    train_data_X =[]
    train_data_Y =[]
    train_length =[]

    #这段代码是为了将训练数据整理成适合深度学习模型训练的形式
    #遍历训练集中每个视频标识及其对应的数据。
    for key,value in train_video_mapping.items():

        #将每个视频的帧索引和对应的值（在这里是训练数据的索引）按列堆叠在一起。这是为了后续对数据的排序。
        lst = np.column_stack((train_video_mapping_index[key],value)  )
        #按照帧索引对数据进行排序，以确保它们按时间顺序排列。
        ind = np.asarray(sorted(lst,key=lambda x: x[0]))

        #创建两个空列表，用于存储训练数据的特征和标签。
        lst_X, lst_Y=[],[]
        #初始化计数器，用于记录每个视频的帧数。
        ctr=0;
        #遍历排序后的数据。
        for i in range(ind.shape[0]):
            #对计数器进行递增，表示当前视频的帧数。
            ctr+=1

            #lst_X.append(preprocessing.scale( min_max_scaler.fit_transform(data_train[train_indices[ind[i,1]]][:-1])))

            #将当前帧的特征添加到 lst_X 中。这里使用了 train_indices 字典将视频的索引映射为实际的训练数据索引。
            lst_X.append(data_train[train_indices[ind[i,1]]][:-1])
            #将当前帧的标签添加到 lst_Y 中。
            lst_Y.append(data_train[train_indices[ind[i,1]]][-1])
        #将当前视频的帧数添加到 train_length 列表中。
        train_length.append(ctr)
        #对于视频的余下帧数（如果存在不足 max_len 的情况）：
        for i in range(ctr, max_len):
            #用之前定义的 pad 进行填充。
            lst_X.append(pad)
            #添加一个虚拟的标签（这里是 0，可以根据实际情况进行调整）。
            lst_Y.append(0)

        #将整理好的视频数据的特征和标签添加到对应的列表中。
        train_data_X.append(lst_X)
        train_data_Y.append(lst_Y)
    

    test_data_X =[]
    test_data_Y =[]
    test_length =[]

    print ("Mapping test")

    #同上
    for key,value in test_video_mapping.items():

        lst = np.column_stack((test_video_mapping_index[key],value)  )
        ind = np.asarray(sorted(lst,key=lambda x: x[0]))

        lst_X, lst_Y=[],[]
        ctr=0
        for i in range(ind.shape[0]):
            ctr+=1
            #lst_X.append(preprocessing.scale( min_max_scaler.transform(data_test[test_indices[ind[i,1]]][:-1])))
            lst_X.append(data_test[test_indices[ind[i,1]]][:-1])
            lst_Y.append(data_test[test_indices[ind[i,1]]][-1])
        test_length.append(ctr)
        for i in range(ctr, max_len):
            lst_X.append(pad)
            lst_Y.append(0) #dummy label

        test_data_X.append(np.asarray(lst_X))
        test_data_Y.append(np.asarray(lst_Y))

    #这部分代码主要是将整理好的训练数据和测试数据转换成NumPy数组，并输出它们的形状以及帧数信息。
    #将训练数据和测试数据的特征列表转换为NumPy数组。这是因为深度学习框架通常使用NumPy数组或类似的数据结构来表示输入数据。
    train_data_X = np.asarray(train_data_X)
    test_data_X = np.asarray(test_data_X)
    #输出训练数据和测试数据的形状，以及它们对应的视频帧数。具体解释如下：
    #输出训练数据特征的形状，即 (视频数量, 帧数, 特征维度)。
    #输出测试数据特征的形状，同样是 (视频数量, 帧数, 特征维度)。
    #输出训练数据中每个视频的帧数。
    #输出测试数据中每个视频的帧数。
    print (train_data_X.shape, test_data_X.shape,len(train_length), len(test_length))

    #这部分代码是将整理好的数据保存到 pickle 文件中。
    #输出提示信息，表示即将将数据保存到文件中。
    print ("Dumping data")
    #使用二进制写入模式 ('wb') 打开一个文件，文件路径是 ./input/ 目录下，文件名是 name 变量的值，后缀是 .pickle。
    with open('./input/'+name+'.pickle', 'wb') as handle:
        #使用 pickle 库的 dump 函数将数据保存到文件中。具体保存的数据包括：
        pickle.dump((train_data_X,  np.asarray(train_data_Y), test_data_X, np.asarray(test_data_Y), max_len ,train_length, test_length), handle, protocol=pickle.HIGHEST_PROTOCOL)
        #train_data_X：训练数据的特征。
        #np.asarray(train_data_Y)：训练数据的标签（类别）。
        #test_data_X：测试数据的特征。
        #np.asarray(test_data_Y)：测试数据的标签（类别）。
        #max_len：视频帧的最大长度。
        #train_length：训练数据中每个视频的帧数。
        #test_length：测试数据中每个视频的帧数。


    


if __name__ == "__main__":

    names = ['text','audio','video']
    for nm in names:
        main(nm)