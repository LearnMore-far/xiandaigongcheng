# -*- coding: utf-8 -*-
'''
从原数据中选取部分数据；
选取数据的title前两个字符在字典WantedClass中；
且各个类别的数量为WantedNum
'''
import jieba
import json


DataJsonFile = r'data\data.json'
MyTainJsonFile = r'data\my_train_data.json'
MyValidJsonFile = r'data\my_valid_data.json'
StopWordFile = r'data/stopword.txt'

WantedClass = {'教育': 0, '健康': 0, '生活': 0, '娱乐': 0, '游戏': 0}
TrainNum = 1000
numTrainAll = TrainNum * 5
numValidAll = 1000


def main():
    Datas = open(DataJsonFile, 'r', encoding='utf_8').readlines()
    train_f = open(MyTainJsonFile, 'w', encoding='utf_8')
    valid_f = open(MyValidJsonFile, 'w', encoding='utf_8')

    TrainInWanted = 0
    ValidInWanted = 0
    for line in Datas:
        data = json.loads(line)
        cla = data['category'][0:2]
        if cla in WantedClass and WantedClass[cla] < TrainNum and TrainInWanted < numTrainAll:
            json_data = json.dumps(data, ensure_ascii=False)
            train_f.write(json_data)
            train_f.write('\n')
            WantedClass[cla] += 1
            TrainInWanted += 1
        elif cla in WantedClass and ValidInWanted < numValidAll:
            json_data = json.dumps(data, ensure_ascii=False)
            valid_f.write(json_data)
            valid_f.write('\n')
            ValidInWanted += 1
            if ValidInWanted >= numValidAll:
                break


if __name__ == '__main__':
    main()