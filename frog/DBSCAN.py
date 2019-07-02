import numpy as np
import pandas as pd
import random
import kMeans

def DBSCAN(data, eps, minPts):
    traindata = data.drop(['Species'], axis=1).values
    trainlabel = data['Species'].values
    n_eps = {}
    omega = set([])
    k_group = []
    unvisited = set(range(len(traindata)))
    w = 0
    for i in range(len(traindata)):
        n_eps[i] = []
    for i in range(len(traindata)):
        for j in range(i+1, len(traindata)):
            train_i = traindata[i]
            train_j = traindata[j]
            print((i,j))
            if np.linalg.norm(train_i-train_j) <= eps:
                n_eps[i].append(j)
                n_eps[j].append(i)
    for key in range(len(traindata)):
        n_eps[key] = set(n_eps[key])
        if len(n_eps[key]) > minPts:
            omega.add(key)
    while len(omega) > 0:
        o = random.sample(omega, 1)[0]
        omega_curr = {o}
        k_new = {o}
        unvisited -= {o}
        w = w + 1
        print(len(omega))
        while len(omega_curr) > 0:
            o2 = random.sample(omega_curr, 1)[0]
            delta = n_eps[o2] & unvisited
            k_new = k_new | delta
            unvisited = unvisited - delta
            omega_curr = (omega_curr | (delta & omega)) - {o2}
            print(len(omega_curr))
            # print(len(omega_curr))
        k_group.append(k_new)
        omega = omega - k_new
    # print(len(k_group))
    k_label = []
    for i in range(len(trainlabel)):
        for j in range(len(k_group)):
            if i in k_group[j]:
               k_label.append(j)
               break
    save_label = k_label.copy()
    save_label.insert(0, len(k_group))
    pd.DataFrame(save_label) \
        .to_csv('PCA.csv', header=0, index=False, sep=',')
    print(kMeans.purity(k_label, trainlabel), kMeans.calculd(k_label, trainlabel))
    return k_group

if __name__ == '__main__':
    raw_data = pd.DataFrame(pd.read_csv('Frogs_MFCCs.csv'))
    raw_data = raw_data.drop(['RecordID'], axis=1)
    raw_data = raw_data.drop(['Family'], axis=1)
    raw_data = raw_data.drop(['Genus'], axis=1)
    DBSCAN(raw_data, 0.4, 150)