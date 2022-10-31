import random

import numpy as np
import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


# from collections import Counter


# 取5000条数据
def write_file():
    pd_all = pd.read_csv("data\\weibo_senti_100k.csv", nrows=80000)
    sen = pd_all.values.tolist()
    label = []
    review = []
    for i in sen[6000:6100]:
        label.append(i[0])
        review.append(i[1])
    for i in sen[73000:73100]:
        label.append(i[0])
        review.append(i[1])
    print(len(label), sum(label))
    # 字典中的key值即为csv中列名
    dataframe = pd.DataFrame({'label': label, 'review': review})
    # 将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv("data\\weibo_senti_test_200.csv", index=False, sep=',')


# 加载数据
def load_data(path, nrows):
    """
    :param path: Path of the CSV file to be loaded (string)
    :return: pos/nag labels (list)
            review (list)
            pos_number (int)
            bag_number (int)
    """
    pd_all = pd.read_csv(path, nrows=nrows)
    pos_number = pd_all[pd_all.label == 1].shape[0]
    nag_number = pd_all[pd_all.label == 0].shape[0]
    sen = pd_all.values.tolist()
    label = []
    review = []
    for i in sen:
        label.append(i[0])
        review.append(i[1])
    return label, review, pos_number, nag_number


# 按行读取文件，返回文件的行字符串列表
def read_file(file_name):
    fp = open(file_name, "r", encoding="utf-8-sig")
    content_lines = fp.readlines()
    fp.close()
    # 去除行末的换行符，否则会在停用词匹配的过程中产生干扰
    for i in range(len(content_lines)):
        content_lines[i] = content_lines[i].rstrip("\n")
    return content_lines


# 分词，剔除停用词
def delete_stopwords(lines):
    """
    :param lines: 代分词集合 (list)
    :return: 分词完成后的集合 (list) 以及 词频统计 (dict)
    """
    stopwords = read_file("data\\stopwords.dat")
    words = []
    all_words = []
    for line in lines:
        temp = [word for word in jieba.cut(line) if word not in stopwords]
        words.append(temp)
        all_words += temp

    # dict_words = dict(Counter(all_words))

    return words


# 特征提取，计算tfidf
def FeatureExtraction(corpus):
    """
    :param corpus: Participles collection (list)
    :return: tfidf matrix (numpy.matrix)
    """
    # 统计词频
    vectorizer = CountVectorizer()
    csr_mat = vectorizer.fit_transform(corpus)
    # 特征提取，计算tfidf
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(csr_mat)
    return tfidf.todense()


# 构建数据集tfidf
def createDataSet(path="data\\weibo_senti_200.csv", number=200):
    np.random.seed(1)  # 设置一个固定的随机种子，以保证接下来的步骤中我们的结果是一致的。

    label, reviews, pn, nn = load_data(path, number)
    print("积极评论样本数：", pn)
    print("消极评论样本数：", nn)
    Y = label
    words = delete_stopwords(reviews)
    temp = []
    for i in words:
        temp.append(" ".join(i))
    X = FeatureExtraction(temp)
    X = X.tolist()
    # 拆分训练集与测试集下标
    trainSet = list(range(pn + nn))
    testSet = []
    for i in range(int((pn + nn) * 0.2)):
        randIndex = int(random.uniform(0, len(trainSet)))
        testSet.append(trainSet[randIndex])
        del (trainSet[randIndex])
    # 训练矩阵
    trainMat = []
    # 分类表
    trainClass = []
    # 生成训练集
    for docIndex in trainSet:
        trainMat.append(X[docIndex])
        trainClass.append(Y[docIndex])
    trainMat = np.array(trainMat).reshape((len(trainMat[0]), len(trainMat)))
    trainClass = np.array(trainClass).reshape((1, len(trainClass)))
    print("trainMat的维度为: " + str(trainMat.shape))
    print("trainClass的维度为: " + str(trainClass.shape))
    print("训练数据集里面的数据有：" + str(trainClass.shape[1]) + " 个")
    testMat = []
    testClass = []
    # 生成训练集
    for docIndex in testSet:
        testMat.append(X[docIndex])
        testClass.append(Y[docIndex])
    testMat = np.array(testMat).reshape((len(testMat[0]), len(testMat)))
    testClass = np.array(testClass).reshape((1, len(testClass)))

    print("testMat维度是: " + str(testMat.shape))
    print("testClass维度是: " + str(testClass.shape))
    print("测试数据集里面的数据有：" + str(testClass.shape[1]) + " 个")

    return trainMat, trainClass, testMat, testClass


# 语料表
def createVocabList(doclist):
    vocabSet = set([])
    for doc in doclist:
        vocabSet = vocabSet | set(doc)
    return list(vocabSet)


# 文章向量化
def setOfWord2Vec(vocablist, inputSet):
    returnVec = [0] * len(vocablist)
    for word in inputSet:
        if word in vocablist:
            returnVec[vocablist.index(word)] = 1
    return returnVec


# onehot编码
def createDatasetOnehot(path="data\\weibo_senti_200.csv", number=200):
    np.random.seed(1)
    label, reviews, pn, nn = load_data(path, number)
    print("训练集积极评论样本数：", pn)
    print("训练集消极评论样本数：", nn)
    doclist = delete_stopwords(reviews)
    vocablist = createVocabList(doclist)
    # 训练矩阵
    trainMat = []
    # 分类表
    trainClass = []
    # 拆分训练集与测试集下标
    # trainSet = list(range(pn + nn))
    # testSet = []
    # for i in range(int((pn + nn) * 0.2)):
    #     randIndex = int(random.uniform(0, len(trainSet)))
    #     testSet.append(trainSet[randIndex])
    #     del (trainSet[randIndex])
    # 生成训练集
    for docInedx in range(pn + nn):
        trainMat.append(setOfWord2Vec(vocablist, doclist[docInedx]))
        trainClass.append(label[docInedx])
    trainMat = np.array(trainMat).reshape((len(trainMat[0]), len(trainMat)))
    trainClass = np.array(trainClass).reshape((1, len(trainClass)))
    print("trainMat的维度为: " + str(trainMat.shape))
    print("trainClass的维度为: " + str(trainClass.shape))
    print("训练数据集里面的数据有：" + str(trainClass.shape[1]) + " 个")

    label_, reviews_, pn_, nn_ = load_data("data\\weibo_senti_test_200.csv", 240)
    print("测试集积极评论样本数：", pn_)
    print("测试集消极评论样本数：", nn_)
    doclist_ = delete_stopwords(reviews_)
    testMat = []
    testClass = []
    for docIdx in range(pn_ + nn_):
        testMat.append(setOfWord2Vec(vocablist, doclist_[docIdx]))
        testClass.append(label_[docIdx])
    testMat = np.array(testMat).reshape((len(testMat[0]), len(testMat)))
    testClass = np.array(testClass).reshape((1, len(testClass)))
    print("testMat的维度为: " + str(testMat.shape))
    print("testClass的维度为: " + str(testClass.shape))
    print("测试数据集里面的数据有：" + str(testClass.shape[1]) + " 个")
    return trainMat, trainClass, testMat, testClass


if __name__ == '__main__':
    write_file()
