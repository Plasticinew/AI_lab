import pandas as pd
import numpy as np
import heapq
from collections import Counter


def accuracy(predictlabel,testlabel):
    count = 0
    for i in range(len(predictlabel)):
        if predictlabel[i] == testlabel[i]:
            count = count+1
    return count/len(predictlabel)


def F1Score(predictlabel, testlabel, true):
    TP = 0
    FP = 0
    FN = 0
    RN = 0
    for i in range(len(predictlabel)):
        if predictlabel[i] == true and testlabel[i] == true:
            TP = TP+1
        elif predictlabel[i] == true:
            FP = FP+1
        elif testlabel[i] != true:
            FN = FN+1
        else:
            RN = RN+1
    P = TP/(TP+FP)
    R = TP/(TP+FN)
    return 2 * P * R / (P + R), TP, FP, FN, RN


def macroF1(predictlabel,testlabel):
    Score=[]
    for true in set(predictlabel):
        Score.append(F1Score(predictlabel, testlabel,true)[0])
    return np.average(Score)


def microF1(predictlabel,testlabel):
    TP = 0
    FP = 0
    FN = 0
    for true in set(predictlabel):
        zero, tp, fp, fn, rn = F1Score(predictlabel, testlabel, true)
        TP = TP + tp
        FP = FP + fp
        FN = FN + fn
    P = TP / (TP + FP)
    R = TP / (TP + FN)
    return 2 * P * R / (P + R)


def euclideanDistances(test, trainset, k):
    testdata = np.array(test)
    distance = []
    heapq.heapify(distance)
    count = 0
    for train in trainset.values:
        train = np.array(train)
        value = np.linalg.norm(train-testdata)
        # np.sqrt(((train - testdata) ** 2).sum())
        if len(distance) < k or -distance[0][0] > value:
            heapq.heappush(distance, (value, count))
        count += 1
    return distance


def kNear(distance, trainlabel, k):
    label = []
    for i in range(k):
        label.append(trainlabel[distance[i][1]])
    return Counter(label).most_common(1)[0][0]


def knn(trainset, trainlabel, testset, testlabel, k):
    ypred = []
    i = 0
    for test in testset.values:
        distance = euclideanDistances(test, trainset, k)
        label = kNear(distance, trainlabel, k)
        if i % 100 == 0:
            print('%d,%s', [i, label])
        i = i+1
        ypred.append(label)
    print(accuracy(ypred, testlabel))
    print(macroF1(ypred, testlabel))
    print(microF1(ypred, testlabel))
    return ypred


if __name__ == '__main__':
    trainset = pd.DataFrame(pd.read_csv('trainset.csv'))
    trainlabel = trainset['6']
    trainset = trainset.drop(['6'], axis=1)
    for index in ('0', '2', '4'):
        trainset[index] = trainset[index].replace(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'], [1, 2, 3, 4, 5, 6, 7, 8])
    testset = pd.DataFrame(pd.read_csv('testset.csv'))
    testlabel = testset['6']
    testset = testset.drop(['6'], axis=1)
    for index in ('0', '2', '4'):
        testset[index] = testset[index].replace(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'], [1, 2, 3, 4, 5, 6, 7, 8])
    ypred=knn(trainset,trainlabel,testset,testlabel, 7)
    #print(microF1(['a','b','c','b','c'],['a','c','b','b','c']))
    print(trainset)
    print(trainlabel)


