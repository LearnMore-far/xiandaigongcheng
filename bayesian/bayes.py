
from load_data import createDatasetOnehot
import numpy as np


def evaluationIndex(predictions, testClass):
    """
    :param predictions: 预测值 (1, 测试样本数)
    :param testClass:  实际值 (1, 测试样本数)
    :return:
    """
    # 正类预测为正类
    tp = 0
    # 正类预测为负类
    fn = 0
    # 负类预测为负类
    tn = 0
    # 负类预测为正类
    fp = 0
    for i in range(testClass.shape[1]):
        if predictions[0, i] == 1 and testClass[0, i] == 1:
            tp += 1
        elif predictions[0, i] == 1 and testClass[0, i] == 0:
            fn += 1
        elif predictions[0, i] == 0 and testClass[0, i] == 0:
            tn += 1
        elif predictions[0, i] == 0 and testClass[0, i] == 1:
            fp += 1
    print("tp ", tp, "fn ", fn, "tn ", tn, "fp ", fp)
    # 准确率
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    print("accuracy：", accuracy * 100, "%")
    # 召回率
    recall = tp / (tp + fn)
    print("recall：", recall * 100, "%")
    # 精确率
    precision = tp / (tp + fp)
    print("precision：", precision * 100, "%")
    # F值
    F = precision * recall / (2 * (precision + recall))
    print("F1 score：", F)


# 训练函数
def trainNB(trainMat, trainClass):
    """
    :param trainMat: (m, nx)
    :param trainClass: (m,)
    :return:
    """
    # 总训练样本数
    numTrainDocs = trainMat.shape[0]
    # 向量维度
    numWords = trainMat.shape[1]
    # 积极评论概率
    p1 = sum(trainClass) / float(numTrainDocs)
    p0Num = np.ones(numWords)  # 拉普拉斯平滑
    p1Num = np.ones(numWords)
    p0Denom = 2
    p1Denom = 2
    for docIndex in range(numTrainDocs):
        if trainClass[docIndex] == 1:
            # 分子
            p1Num += trainMat[docIndex]
            # 分母
            p1Denom += sum(trainMat[docIndex])
        else:
            p0Num += trainMat[docIndex]
            p0Denom += sum(trainMat[docIndex])
    p0Vec = np.log(p0Num / p0Denom)
    p1Vec = np.log(p1Num / p1Denom)
    return p0Vec, p1Vec, p1


# 预测模块
def classifyNB(wordVec, p0Vec, p1Vec, p1_class):
    p0 = np.log(1 - p1_class) + sum(wordVec * p0Vec)
    p1 = np.log(p1_class) + sum(wordVec * p1Vec)
    if p1 > p0:
        return 1
    return 0


def classifyDoc():
    trainMat, trainClass, testMat, testClass = createDatasetOnehot(path="data\\weibo_senti_200.csv", number=200)
    p0Vec, p1Vec, p1 = trainNB(trainMat.T, trainClass.reshape((trainClass.shape[1],)))
    predictions = []
    for docIndex in range(testMat.shape[1]):
        pre = classifyNB(testMat.T[docIndex, :], p0Vec, p1Vec, p1)
        predictions.append(pre)
    evaluationIndex(np.array(predictions).reshape((1, len(predictions))), testClass)


if __name__ == '__main__':
    classifyDoc()
