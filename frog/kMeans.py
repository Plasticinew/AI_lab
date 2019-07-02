import pandas as pd
import numpy as np
from collections import Counter


def purity(k_label, train_label):
    counter = {}
    sum_value = 0
    for vec in range(len(k_label)):
        train = train_label[vec]
        k = k_label[vec]
        if k not in counter.keys():
            counter[k] = {}
        if train not in counter[k].keys():
            counter[k][train] = 0
        counter[k][train] += 1
    for group in counter.values():
        sum_value += max(list(group.values()))
    return sum_value / len(k_label)


def calculd(k_label, train_label):
    a = 0
    d = 0
    c2n = 0
    for vec1 in range(len(k_label)):
        for vec2 in range(vec1+1, len(k_label)):
            if train_label[vec1] == train_label[vec2]:
                if k_label[vec1] == k_label[vec2]:
                    a += 1
            elif k_label[vec1] != k_label[vec2]:
                d += 1
            c2n += 1
    return (a + d)/c2n


def distance(x, y):
    return np.linalg.norm(y-x)


def nextk(traindata, k_center):
    k_group = {}
    new_k_center = []
    k_group_label = []
    for vec in traindata:
        min_value = 1000
        min_group = 1000
        for i in range(len(k_center)):
            value = distance(k_center[i], vec)
            if value < min_value:
                min_value = value
                min_group = i
        if min_group not in k_group.keys():
            k_group[min_group] = []
        k_group[min_group].append(vec)
        k_group_label.append(min_group)
    for group in k_group.values():
        new_k_center.append(np.mean(group, 0))
    is_next = False
    if len(k_center) != len(new_k_center):
        is_next = True
    for vec in k_center:
        is_in = False
        for new_vec in new_k_center:
            if distance(new_vec, vec) < 0.00001:
                is_in = True
                break
        if not is_in:
            is_next = True
            break
    if is_next:
        return nextk(traindata, new_k_center)
    else:
        return new_k_center, k_group_label


def kMeans(k, data):
    traindata = data.drop(['Species'], axis=1).values
    trainlabel = data['Species'].values
    vec_num = len(traindata[0])
    k_center = np.random.rand(k, vec_num)
    k_center, k_label = nextk(traindata, k_center)
    save_label = k_label.copy()
    print(len(k_center))
    save_label.insert(0, len(k_center))
    pd.DataFrame(save_label)\
        .to_csv('kMeans.csv', header=0, index=False, sep=',')
    print(purity(k_label, trainlabel), calculd(k_label, trainlabel))
    # raw_data.to_csv('kMeans.csv', index=False, sep=',')

if __name__ == '__main__':
    raw_data = pd.DataFrame(pd.read_csv('Frogs_MFCCs.csv'))
    raw_data = raw_data.drop(['RecordID'], axis=1)
    raw_data = raw_data.drop(['Family'], axis=1)
    raw_data = raw_data.drop(['Genus'], axis=1)
    kMeans(10, raw_data)
