import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import kMeans


def paint(data, label):
    marker = ['x', '+', 'o']
    group = {}
    for i in range(len(data)):
        data_value = data[i]
        label_value = label[i]
        if label_value not in group.keys():
            group[label_value] = []
        group[label_value].append(data_value)
    count = 0
    for g in group.values():
        g = np.array(g)
        plt.scatter(g[:, 0], g[:, 1], label=str(count))
        count += 1
    plt.legend(loc='upper right')
    plt.show()


def PCA(data, threshold):
    traindata = data.drop(['Species'], axis=1).values
    trainlabel = data['Species'].values
    mean_val = np.mean(traindata, 0)
    zero_data = traindata - mean_val
    cov_data = np.cov(zero_data, rowvar=0)
    eig_vals, eig_vects = np.linalg.eig(cov_data)
    m = 0
    for i in range(1, len(eig_vals)):
        if sum(eig_vals[:i-1]) / sum(eig_vals) < \
                threshold <= sum(eig_vals[:i] / sum(eig_vals)):
            m = i
            break
    print(m)
    eig_val_sorted = np.argsort(eig_vals)
    m_eig_val = eig_val_sorted[-1:-(m+1):-1]
    m_eig_vect = eig_vects[:, m_eig_val]
    low_data = np.dot(zero_data, m_eig_vect)
    vec_num = len(low_data[0])
    k_center = np.random.rand(10, vec_num)
    k_center, k_label = kMeans.nextk(low_data, k_center)
    save_label = k_label.copy()
    print(len(k_center))
    save_label.insert(0, len(k_center))
    pd.DataFrame(save_label) \
        .to_csv('PCA.csv', header=0, index=False, sep=',')
    paint(low_data, k_label)
    print(kMeans.purity(k_label, trainlabel), kMeans.calculd(k_label, trainlabel))


if __name__ == '__main__':
    raw_data = pd.DataFrame(pd.read_csv('Frogs_MFCCs.csv'))
    raw_data = raw_data.drop(['RecordID'], axis=1)
    raw_data = raw_data.drop(['Family'], axis=1)
    raw_data = raw_data.drop(['Genus'], axis=1)
    PCA(raw_data, 0.5)

