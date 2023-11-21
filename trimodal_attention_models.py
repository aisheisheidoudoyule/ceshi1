import gc, numpy as np, pickle
import tensorflow as tf
from keras.models import Model
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, History
from keras.layers import Input, Bidirectional, GRU, Masking, Dense, Dropout, TimeDistributed, Lambda, Activation, dot, multiply, concatenate
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.python.keras.callbacks import History


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess=tf.compat.v1.Session(config=config)

def calc_test_result(result, test_label, test_mask, print_detailed_results=True):
    '''
    # Arguments
        predicted test labels, gold test labels and test mask

    # Returns
        accuracy of the predicted labels
    '''
    # print("预测标签---------------------------------------------")
    # print(result)

    true_label=[]                   #真实标签
    predicted_label=[]              #预测标签

    #嵌套循环遍历每个元素，根据掩码来判断是否需要考虑该位置
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            if test_mask[i,j]==1:
                true_label.append(np.argmax(test_label[i,j] ))#将最大test_label【i，j】添加到真实标签中
                predicted_label.append(np.argmax(result[i,j] ))#将。。result。。。添加到预测标签中

    if print_detailed_results:
        print ("Confusion Matrix :")
        print (confusion_matrix(true_label, predicted_label))           #输出混沌矩阵
        print ("Classification Report :")
        print (classification_report(true_label, predicted_label))
    print ("Accuracy ", accuracy_score(true_label, predicted_label))
    return accuracy_score(true_label, predicted_label)



def create_one_hot_labels(train_label, test_label):

    maxlen = int(max(train_label.max(), test_label.max()))
    
    train = np.zeros((train_label.shape[0], train_label.shape[1], maxlen+1))
    test = np.zeros((test_label.shape[0], test_label.shape[1], maxlen+1))
    
    for i in range(train_label.shape[0]):
        for j in range(train_label.shape[1]):
            train[i,j,train_label[i,j]] = 1

    for i in range(test_label.shape[0]):
        for j in range(test_label.shape[1]):
            test[i,j,test_label[i,j]] = 1

    return train, test


def create_mask(train_data, test_data, train_length, test_length, attention_weights=None):          #掩码和其对应的矩阵形式是相同的，但掩码的第一位为1代表所对应的矩阵第一位为有效值，0即为无效值

    train_mask = np.zeros((train_data.shape[0], train_data.shape[1]), dtype='float')
    for i in range(len(train_length)):
        if attention_weights is None:
            attention_mask = np.ones(train_length[i])
        else:
            attention_mask = attention_weights[i][:train_length[i]]

        train_mask[i, :train_length[i]] = attention_mask

    test_mask = np.zeros((test_data.shape[0], test_data.shape[1]), dtype='float')
    for i in range(len(test_length)):
        if attention_weights is None:
            attention_mask = np.ones(test_length[i])
        else:
            attention_mask = attention_weights[i][:test_length[i]]

        test_mask[i, :test_length[i]] = attention_mask

    # print("训练掩码维度:")
    # print(train_mask.shape)
    # print("测试掩码维度")
    # print(test_mask.shape)

    return train_mask, test_mask


(train_text, train_label, test_text, test_label, max_utt_len, train_len, test_len) = pickle.load(open('./input/text.pickle', 'rb'))
(train_audio, _, test_audio, _, _, _, _) = pickle.load(open('./input/audio.pickle', 'rb'))  #载入信息，训练音频信息，标签信息，测试。。，测试。。，最大对话长度，训练长度
(train_video, _, test_video, _, _, _, _) = pickle.load(open('./input/video.pickle', 'rb'))

#测111231231312
# print("载入的信息维度，文本，音频，视频:")
# print(train_text.shape)
# print(train_audio.shape)
# print(train_video.shape)

train_label, test_label = create_one_hot_labels(train_label.astype('int'), test_label.astype('int'))    #将lable转为独热编码形式

train_mask, test_mask = create_mask(train_text, test_text, train_len, test_len)     #生成掩码，标记真实值


num_train = int(len(train_text)*0.8)        #计算了训练集的样本数量，并将其存储在num_train变量中，该数量为整个训练集的80%。

train_text, dev_text = train_text[:num_train, :, :], train_text[num_train:, :, :]
train_audio, dev_audio = train_audio[:num_train, :, :], train_audio[num_train:, :, :]
train_video, dev_video = train_video[:num_train, :, :], train_video[num_train:, :, :]
train_label, dev_label = train_label[:num_train, :, :], train_label[num_train:, :, :]
train_mask, dev_mask = train_mask[:num_train, :], train_mask[num_train:, :]         #用切割，使数据分为训练数据和验证数据，其中训练为80%，验证为20%


#创新点3---------------------------------------------------------------------------------------------------------
def bi_modal_attention(x, y):       #双注意力机制，双模态

     
    m1 = dot([x, y], axes=[2, 2])               #x和y两个矩阵进行乘法，m1的维度是（x的第一维度，x的第二维度，y的第一维度，y的第二维度）
    n1 = Activation('softmax')(m1)              #激活函数可改
    o1 = dot([n1, y], axes=[2, 1])              #同上上
    a1 = multiply([o1, x])                      #o1和x进行逐个元素相乘，要求两个张量形状形同

    m2 = dot([y, x], axes=[2, 2])
    n2 = Activation('softmax')(m2)
    o2 = dot([n2, x], axes=[2, 1])
    a2 = multiply([o2, y])

    #测试
    # return multiply([a1,a2])
    #原版
    return concatenate([a1, a2])
    # return a1
#创新点3结束------------------------------------------------------------------------------------------------------

def self_attention(x):              #自注意力机制，单模态


    m = dot([x, x], axes=[2,2])
    n = Activation('softmax')(m)
    o = dot([n, x], axes=[2,1])
    a = multiply([o, x])

    return a
    

def contextual_attention_model(mode):       #上下文注意力机制
    
    ########### Input Layer ############        输入层
        
    in_text = Input(shape=(train_text.shape[1], train_text.shape[2]))       #shape【1】为文本的长度，shape【2】为单词表示的通道数
    in_audio = Input(shape=(train_audio.shape[1], train_audio.shape[2]))    #shape[1]音频的时间步数，shape[2]每个时间步的特征维度
    in_video = Input(shape=(train_video.shape[1], train_video.shape[2]))    #同音频

    # print("输入层文本模态维度：")
    # print(in_text.shape)
    # print("输入层音频模态维度：")
    # print(in_audio.shape)
    # print("输入层视频模态维度：")
    # print(in_video.shape)
        
    ########### Masking Layer ############      掩码层
        
    masked_text = Masking(mask_value=0)(in_text)            #创建掩码，使其识别有效值，防止浪费资源
    masked_audio = Masking(mask_value=0)(in_audio)
    masked_video = Masking(mask_value=0)(in_video)

    # print("掩码层文本模态维度：")
    # print(masked_text.shape)
    # print("掩码层音频模态维度：")
    # print(masked_audio.shape)
    # print("掩码层视频模态维度：")
    # print(masked_video.shape)

#创新点2-------------------------------------------------------------------
    ########### Recurrent Layer ############                循环层
        
    drop_rnn = 0.7      #定义了在RNN层中应用的dropout比例，即在训练过程中随机丢弃部分神经元以减少过拟合。
    gru_units = 300     #定义了GRU层中的单元数，即每个时间步的输出维度
            
    rnn_text = Bidirectional(GRU(gru_units, return_sequences=True, dropout=0.5, recurrent_dropout=0.5), merge_mode='concat')(masked_text)
    #将掩码文本数据传递给双向GRU层进行处理，获取每个时间步的输出序列。这样可以利用双向上下文信息和时序关系来丰富文本数据的表示。
    rnn_audio = Bidirectional(GRU(gru_units, return_sequences=True, dropout=0.5, recurrent_dropout=0.5), merge_mode='concat')(masked_audio)
    rnn_video = Bidirectional(GRU(gru_units, return_sequences=True, dropout=0.5, recurrent_dropout=0.5), merge_mode='concat')(masked_video)        
            
    rnn_text = Dropout(drop_rnn)(rnn_text)      #随机丢弃，减少过拟合，提升泛华能力和鲁棒性
    rnn_audio = Dropout(drop_rnn)(rnn_audio)
    rnn_video = Dropout(drop_rnn)(rnn_video)        #可以尝试更改

    # print("循环层文本模态维度：")
    # print(rnn_text.shape)
    # print("循环层音频模态维度：")
    # print(rnn_audio.shape)
    # print("循环层视频模态维度：")
    # print(rnn_video.shape)
#创新点2------------------------------------------------------------------------


    ########### Time-Distributed Dense Layer ############       时间分布式稠密层        减少过拟合

    drop_dense = 0.7
    dense_units = 100           #指定了该层的输出维度

    dense_text = Dropout(drop_dense)(TimeDistributed(Dense(dense_units, activation='tanh'))(rnn_text))      #表示激活函数为双曲正切函数
    dense_audio = Dropout(drop_dense)(TimeDistributed(Dense(dense_units, activation='tanh'))(rnn_audio))
    dense_video = Dropout(drop_dense)(TimeDistributed(Dense(dense_units, activation='tanh'))(rnn_video))

    #创新点2
    _no_RNN_dense_text = Dropout(drop_dense)(TimeDistributed(Dense(dense_units, activation='tanh'))(masked_text))
    _no_RNN_dense_audio = Dropout(drop_dense)(TimeDistributed(Dense(dense_units, activation='tanh'))(masked_audio))
    _no_RNN_dense_video = Dropout(drop_dense)(TimeDistributed(Dense(dense_units, activation='tanh'))(masked_video))
    # print("时间分布稠密层文本模态维度：")
    # print(dense_text.shape)
    # print("时间分布稠密层音频模态维度：")
    # print(dense_audio.shape)
    # print("时间分布稠密层视频模态维度：")
    # print(dense_video.shape)


    # dense_vt = Dropout(drop_dense)(TimeDistributed(Dense(dense_units, activation='tanh'))(bi_modal_attention(dense_video, dense_text)))
    # dense_av = Dropout(drop_dense)(TimeDistributed(Dense(dense_units, activation='tanh'))(bi_modal_attention(dense_audio, dense_video)))
    # dense_ta = Dropout(drop_dense)(TimeDistributed(Dense(dense_units, activation='tanh'))(bi_modal_attention(dense_text, dense_audio)))

    ########### Attention Layer ############
        
    ## Multi Modal Multi Utterance Bi-Modal attention ##        #双模态注意力机制的应用
    if mode == 'ONE':   #第三个创新点

        # #原版
        # vt_att = bi_modal_attention(dense_video, dense_text)
        # av_att = bi_modal_attention(dense_audio, dense_video)
        # ta_att = bi_modal_attention(dense_text, dense_audio)
        # vv_att = self_attention(dense_video)  # 将视频的密集特征传入
        # tt_att = self_attention(dense_text)  # 激活函数应该为softmax
        # aa_att = self_attention(dense_audio)
        # #print(ta_att.shape)
        # merged = concatenate([vt_att, av_att, ta_att, dense_video,dense_audio,dense_text])
        # #将上述得到的注意力加权的特征（vt_att、av_att、ta_att）与原始的视频特 征（dense_video）、音频特征（dense_audio）和文本特征（dense_text）进行拼接。
        # #print(merged.shape)

        # # 测试1
        # dence_vt = concatenate([dense_video, dense_text])
        # dence_tv = concatenate([dense_text, dense_video])
        # C_tvvt = bi_modal_attention(dence_tv, dence_vt)
        #
        # dence_av = concatenate([dense_audio, dense_video])
        # dence_va = concatenate([dense_video, dense_audio])
        # C_vaav = bi_modal_attention(dence_va, dence_av)
        #
        # dence_at = concatenate([dense_audio, dense_text])
        # dence_ta = concatenate([dense_text, dense_audio])
        # C_taat = bi_modal_attention(dence_ta, dence_at)
        #
        # vv_att = self_attention(dense_video)  # 将视频的密集特征传入
        # tt_att = self_attention(dense_text)  # 激活函数应该为softmax
        # aa_att = self_attention(dense_audio)
        #
        # merged = concatenate([C_tvvt, C_taat, C_vaav, vv_att, tt_att, aa_att])

        # print("注意力层文本+视频模态维度：")
        # print(dence_vt.shape)
        # print("注意力层音频+文本模态维度：")
        # print(dence_at.shape)
        # print("注意力层视频+音频模态维度：")
        # print(dence_av.shape)

        # #测试2
        # ta_att = bi_modal_attention(dense_audio, dense_text)
        # av_att = bi_modal_attention(dense_video, dense_text)
        # vv_att = self_attention(dense_video)  # 将视频的密集特征传入
        # tt_att = self_attention(dense_text)  # 激活函数应该为softmax
        # aa_att = self_attention(dense_audio)
        # # print(ta_att.shape)
        # merged = concatenate([ av_att, ta_att, dense_video,dense_audio,dense_text])
        # #测试2结束

        # #测试3----只利用双模太注意力融合机制与原版对比---验证创新点3
        # ta_att = bi_modal_attention(rnn_audio, rnn_text)
        # av_att = bi_modal_attention(rnn_video, rnn_text)
        # # print(ta_att.shape)
        # merged = concatenate([av_att, ta_att, rnn_video, rnn_audio, rnn_text])
        # #测试3结束

        #测试4----为经过双向循环神经网络对齐后，但为加入双模态注意力融合机制与原版对比----验证创新点2
        merged = concatenate([dense_video, dense_audio, dense_text])
        # 测试4结束

        #测试5----测试5为将创新点2+创新点3；（对照实验为单独的创新点2，和创新点3）
        ta_att = bi_modal_attention(dense_audio, dense_text)
        av_att = bi_modal_attention(dense_video, dense_text)
        # print(ta_att.shape)
        merged = concatenate([av_att, ta_att, dense_video, dense_audio, dense_text])
        #测试5结束


        # 将上述得到的注意力加权的特征（vt_att、av_att、ta_att）与原始的视频特 征（dense_video）、音频特征（dense_audio）和文本特征（dense_text）进行拼接。
        # print(merged.shape)

    ## Multi Modal Uni Utterance Self Attention ##
    elif mode == 'TWO':     #原版
            
        # attention_features = []     #创建一个空列表用来收集，注意力特征
        #
        # for k in range(max_utt_len):
        #
        #     # extract multi modal features for each utterance #
        #     m1 = Lambda(lambda x: x[:, k:k+1, :])(dense_video)  #使用Lambda函数和切片操作可以在循环中逐步提取每个话语的特定时间步特征，用于后续的处理和注意力计算。
        #     m2 = Lambda(lambda x: x[:, k:k+1, :])(dense_audio)
        #     m3 = Lambda(lambda x: x[:, k:k+1, :])(dense_text)
        #
        #     utterance_features = concatenate([m1, m2, m3], axis=1)      #将视频，音频，文字的特征都连接到一起
        #     attention_features.append(self_attention(utterance_features))
        #
        # merged_attention = concatenate(attention_features, axis=1)      #将注意力拼接，按照第一个维度进行拼接，得到整体的注意力特征表示
        # merged_attention = Lambda(lambda x: K.reshape(x, (-1, max_utt_len, 3*dense_units)))(merged_attention)       #将张量进行重新塑形
        #
        # merged = concatenate([merged_attention, dense_video, dense_audio, dense_text])      #将注意力特征，音频特征，视频特征，文字特征进行拼接，形成真正的多模态特征

        # #测试1
        #     vt_att = bi_modal_attention(dense_video, dense_text)
        #     av_att = bi_modal_attention(dense_audio, dense_video)
        #     ta_att = bi_modal_attention(dense_text, dense_audio)
        #     # vv_att = self_attention(dense_video)  # 将视频的密集特征传入
        #     # tt_att = self_attention(dense_text)  # 激活函数应该为softmax
        #     # aa_att = self_attention(dense_audio)
        #     # print(ta_att.shape)
        #     merged = concatenate([ rnn_video, rnn_audio, rnn_text])
        #     # 将上述得到的注意力加权的特征（vt_att、av_att、ta_att）与原始的视频特 征（dense_video）、音频特征（dense_audio）和文本特征（dense_text）进行拼接。
        #     # print(merged.shape)
        # #测试1结束

        # #完全原版--------对应one中的测试3，4
        # merged = concatenate([rnn_video, rnn_audio, rnn_text])
        # #完全原版结束

    #测试3---与one中测试5对应作为对照组，未使用创新2
        ta_att = bi_modal_attention(rnn_audio, rnn_text)
        av_att = bi_modal_attention(rnn_video, rnn_text)
        # print(ta_att.shape)
        merged = concatenate([av_att, ta_att, rnn_video, rnn_audio, rnn_text])
    #

    ## Multi Utterance Self Attention ##        实现了多个话语之间的自注意力机制，用于捕捉不同话语之间的关联信息。
    # elif mode == 'THREE':

        # vv_att = self_attention(dense_video)        #将视频的密集特征传入
        # tt_att = self_attention(dense_text)         #激活函数应该为softmax
        # aa_att = self_attention(dense_audio)
        #
        # merged = concatenate([aa_att, vv_att, tt_att, dense_video, dense_audio, dense_text])
        # #vv_att、tt_att和aa_att分别表示视频、文本和音频的自注意力计算结果。
        # #通过concatenate将它们与视频音频文字的原始特征进行拼接



    ## No Attention ##
    # elif mode == 'None':
    #
    # #     merged = concatenate([dense_video, dense_audio, dense_text])
    #     # 测试3
    #     dence_vt = concatenate([dense_video, dense_text])
    #     dence_tv = concatenate([dense_text, dense_video])
    #     C_tvvt = bi_modal_attention(dence_tv, dence_vt)
    #     dence_tvvt = concatenate([dence_tv, dence_vt])
    #
    #     dence_av = concatenate([dense_audio, dense_video])
    #     dence_va = concatenate([dense_video, dense_audio])
    #     C_vaav = bi_modal_attention(dence_va, dence_av)
    #     dence_vaav = concatenate([dence_av, dence_va])
    #
    #     dence_at = concatenate([dense_audio, dense_text])
    #     dence_ta = concatenate([dense_text, dense_audio])
    #     C_taat = bi_modal_attention(dence_ta, dence_at)
    #     dence_taat = concatenate([dence_ta, dence_at])
    #
    #     A = bi_modal_attention(C_taat, dence_taat)
    #     B = bi_modal_attention(C_tvvt, dence_tvvt)
    #     C = bi_modal_attention(C_vaav, dence_vaav)
    #
    #     merged = concatenate([A, B, C, dense_text, dense_audio, dense_video])
    else:
        print ("模块名出错，请仔细核对，是否为创新“one”和原版“two”")
        return


                
        
    ########### Output Layer ############
        
    output = TimeDistributed(Dense(2, activation='softmax'))(merged)        #建立一个全连接层，merged 是之前定义的特征表示，作为输入传递给输出层。
    model = Model([in_text, in_audio, in_video], output)                    #定义了模型的输入和输入，其中输入为in，输出为output
    # output = str(output)
    # print("output == " + output)
    return model



def train(mode):
    
    runs = 2        #运行的次数，每次运行都会使用不同的随机种子进行训练。
    accuracy = []   #用于存储每次运行的测试准确率。
    loss_values = []
    #测试‘’‘’‘
    accuracy_score = []  # 存储 F1 分数的列表
    #测试’‘’‘’
    for j in range(runs):

        print(str(j)+"     "+mode)
        np.random.seed(j)       #设置种子，使每次运行的结果可以复现
        tf.random.set_seed(j)   #同上
        
        # compile model #
        model = contextual_attention_model(mode)    #调用上面的方法，创建上下文注意力
        model.compile(optimizer='adam', loss='categorical_crossentropy', sample_weight_mode='temporal', metrics=['accuracy'])
        #配置训练过程，optimizer='adam' 指定了使用 Adam 优化器进行模型训练。
        #loss='categorical_crossentropy'使用的是分类交叉熵损失函数。
        #sample_weight_mode='temporal' 指定了样本权重的计算方式。"temporal"，表示样本权重是按时间步进行计算的。
        #metrics=['accuracy'] 指定了模型评估指标，这里使用的是准确率作为评估指标，在训练过程中会计算并输出模型在每个批次上的准确率。

        # # Save model weights
        # model.save_weights('model_weights.h5')
        # from keras.models import load_model
        #
        # # Load the model
        # model = contextual_attention_model(mode)
        # model.load_weights('model_weights.h5')

        # # Print model summary
        # model.summary()

        # set callbacks #       设置回调函数      回调函数用于在训练过程中监控模型的性能并采取相应的操作，例如早停止和保存最佳模型权重。
        path = 'weights/Mosi_Trimodal_' + mode + '_Run_' + str(j) + '.hdf5'

        early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=0)  #如果在连续10个训练周期中都没有观察到损失值下降，则停止训练。
        check = ModelCheckpoint(path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=0)    #保存最好的模型权重

        history = History()
        # train model #
        history = model.fit([train_text, train_audio, train_video], train_label,    #训练数据和标签
                            epochs=15,                      #轮数
                            batch_size=32,                  #每个批次的样本数量，原版batch_size=32
                            sample_weight=train_mask,       #对不同的样本进行加权处理
                            shuffle=False,                   #每个训练周期都“不”对训练数据重新洗牌，保证随机性
                            callbacks=[early_stop, check,history],  #指定了回调参数
                            # validation_data=([dev_text, dev_audio, dev_video], dev_label, dev_mask),
                            validation_data=([test_text, test_audio, test_video], test_label, test_mask),   #指定了验证集，用于检测性能
                            verbose=1)      #用于训练模型的函数

        # # 测试2
        # # 记录每个epoch的准确率
        # accuracy_per_epoch = history.history['val_accuracy']
        #
        # for epoch, acc in enumerate(accuracy_per_epoch, 1):
        #     # 绘制准确率线性变化图
        #     # plt.plot(epoch, acc, 'o', label=f'Run {j + 1}')
        #     plt.plot(epoch, acc, '-o',color = 'blue')
        # plt.xlabel('Epoch')
        # plt.ylabel('Accuracy')
        # plt.title(f'Accuracy Curve for Mode: {mode}')
        # plt.legend()
        #
        # plt.show()
        # # 测试2

        loss_values.append(history.history['loss'])

        # test results #
        model.load_weights(path)        #用于加载之前的权重
        test_predictions = model.predict([test_text, test_audio, test_video])   #用加载的模型对数据进行预测
        test_accuracy = calc_test_result(test_predictions, test_label, test_mask)   #用于计算准确值
        accuracy.append(test_accuracy)  #将准确率添加进列表，用以统计和分析


        
        # release gpu memory #          释放gpu内存
        K.clear_session()
        del model, history
        gc.collect()

        # #测试1
        # # 绘制准确率线性变化图
        # avg_accuracy = sum(accuracy) / len(accuracy)
        # max_accuracy = max(accuracy)
        #
        # # 创建迭代次数或时间步长的列表（横轴）
        # iterations = list(range(1, len(loss_values) + 1))
        #
        # # 绘制准确率线性变化图
        # plt.plot(iterations, accuracy, '-o')
        # plt.xlabel('Iterations')
        # plt.ylabel('Test Accuracy')
        # plt.title('Test Accuracy Curve for Mode: ' + mode)
        # plt.show()
        # #测试1



    # summarize test results #          计算测试准确率的平均值和最大值

    avg_accuracy = sum(accuracy)/len(accuracy)      #表示测试准确率的平均值。
    max_accuracy = max(accuracy)                    #表示测试准确率的最大值。
    
    print ('Mode: ', mode)                          #模型
    print ('Avg Test Accuracy:', '{0:.4f}'.format(avg_accuracy), '|| Max Test Accuracy:', '{0:.4f}'.format(max_accuracy))
    print ('-'*55)


if __name__=="__main__":
    
    for mode in ['ONE', 'TWO']:
        train(mode)