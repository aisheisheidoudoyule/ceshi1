# F:\itsoftware\Anaconda
# -*- coding:utf-8 -*-
# Author = TJL
# date:2020/3/13

import jieba.analyse
import jieba
import os

# 添加专有名词，增加分词力度
jieba.suggest_freq('中国社科院研究生院', True)
jieba.suggest_freq('德国ZF集团', True)
jieba.suggest_freq('技术换市场', True)
jieba.suggest_freq('中央企业', True)
jieba.suggest_freq('工作会议', True)
jieba.suggest_freq('国资委主任', True)

#查看是否存在这个
if not os.path.exists('data/train_corpus/'):os.makedirs('data/train_corpus/')
raw_data_path = 'data/raw_data/'
cut_data_path = 'data/train_corpus/'
stop_word_path = 'data/stop_words.txt' #这里只是样例，可以替换自己的

#停用词处理，通过stopwordslist函数加载停用词列表。停用词是一些在文本分析中通常会被过滤掉的常见词汇，如“的”、“是”等。
def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r').readlines()]
    return stopwords

#cut_word 函数用于对原始语料进行分词处理。通过结巴分词对每个文件中的文本进行分词，并在分词的过程中去除停用词。用空格隔开
def cut_word(raw_data_path, cut_data_path,stop_word_path):
    stopwords = stopwordslist(stop_word_path)
    data_file_list = os.listdir(raw_data_path)
    corpus = []
    temp = 0
    for file in data_file_list:
        with open(raw_data_path + file, 'rb') as f:
            print('处理第{}个原始语料文件'.format(temp + 1))
            temp += 1
            lines = f.readlines()
            for line in lines:
                line=line.decode('utf-8').strip().replace(' ','')
                document_cut = jieba.cut(line, cut_all=False)
                document_cut=[word for word in document_cut if word not in stopwords and len(word.strip())>0]
            # print('/'.join(document_cut))
                result = ' '.join(document_cut)
                corpus.append(result)
        #  print(result)
    with open(cut_data_path + 'corpus.txt', 'w', encoding='utf-8') as f:
        for line in corpus:
            f.write(line+'\n')  # 读取的方式和写入的方式要一致




if __name__ == "__main__":
    cut_word(raw_data_path, cut_data_path,stop_word_path)
