import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
from collections import Counter
from sklearn import tree

decisionNode = dict(boxstyle='sawtooth', fc='0.8')
leafNode = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle='<-')


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
    if P == 0 and R == 0:
        result = 0
    else:
        result = 2 * P * R / (P + R)
    return result, TP, FP, FN, RN


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


def informationEntropy(dataset):
    num_entries = len(dataset)
    label_counts = {}
    '''count num of all labels'''
    for feat_vec in dataset:
        current_label = feat_vec[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    infor_entropy = 0.0
    '''calculate information entropy'''
    for key in label_counts:
        prob = float(label_counts[key])/num_entries
        infor_entropy -= prob * math.log(prob, 2)
    return infor_entropy


def getSubDataset(dataset, axis, value):
    sub_dataset = []
    for feat_vec in dataset:
        if feat_vec[axis] == value:
            sub_dataset.append(feat_vec)
    return sub_dataset


def choosseBestFeature(dataset):
    num_features = len(dataset[0])-1
    init_entropy = informationEntropy(dataset)
    if init_entropy == 0.0:
        return 0
    best_entropy = 0.0
    best_feature = 0
    for i in range(num_features):
        feat_list = [data[i] for data in dataset]
        data_values = set(feat_list)
        new_entropy = 0.0
        for value in data_values:
            sub_dataset = getSubDataset(dataset, i, value)
            new_entropy += len(sub_dataset)/float(len(dataset))*informationEntropy(sub_dataset)
        new_entropy = init_entropy - new_entropy
        if new_entropy > best_entropy:
            best_feature = i
            best_entropy = new_entropy
    if best_entropy < 0.0001:
        best_feature = 0
    return best_feature


def createChildTree(dataset, labels):
    newlabel = labels.copy()
    child_tree = {}
    best_feature = choosseBestFeature(dataset)
    default = Counter([data[-1] for data in dataset]).most_common(1)[0][0]
    if best_feature == 0:
        return default

    child_tree['default'] = default
    child_tree['value'] = labels[best_feature]
    sub_dataset = {}
    for feat_vec in dataset:
        feature = feat_vec[best_feature]
        if feature not in sub_dataset.keys():
            sub_dataset[feature] = []
        sub_dataset[feature].append(np.delete(feat_vec, best_feature))
    newlabel.pop(best_feature)
    for key, sub_data in sub_dataset.items():
        child_tree[key] = createChildTree(sub_data, newlabel)
    return child_tree


def modelTree(tree, testset):
    test_label = []
    for test in testset:
        iterator = tree
        while type(iterator).__name__ == 'dict':
            if test[iterator['value']] not in iterator.keys():
                iterator = iterator['default']
            else:
                iterator = iterator[test[iterator['value']]]
        test_label.append(iterator)
    return test_label


def retrieve_tree(i):
    list_of_trees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                     {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                    ]
    return list_of_trees[i]


def get_num_leafs(tree):
    num_leaves = 0
    for key in tree.keys():
        if type(tree[key]).__name__ == 'dict':
            num_leaves += get_num_leafs(tree[key])
        else:
            num_leaves += 1
    return num_leaves


def get_tree_depth(tree):
    max_depth = 0
    for key in tree.keys():
        if type(tree[key]).__name__ == 'dict':
            this_depth = 1 + get_tree_depth(tree[key])
        else:
            this_depth = 1
        if this_depth > max_depth:
            max_depth = this_depth
    return max_depth


def plot_node(ax, node_txt, center_ptr, parent_ptr, node_type):
    ax.annotate(node_txt, xy=parent_ptr, xycoords='axes fraction',
                xytext=center_ptr, textcoords='axes fraction',
                va='center', ha='center', bbox=node_type, arrowprops=arrow_args)


def plot_mid_text(ax, center_ptr, parent_ptr, txt):
    x_mid = (parent_ptr[0] - center_ptr[0]) / 2.0 + center_ptr[0]
    y_mid = (parent_ptr[1] - center_ptr[1]) / 2.0 + center_ptr[1]
    ax.text(x_mid, y_mid, txt)


def plot_tree(ax, mytree, parent_ptr, node_txt):

    num_leafs = get_num_leafs(mytree)

    first_str = mytree['value']
    center_ptr = (plot_tree.x_off + (1.0 + float(num_leafs)) / 2.0 / plot_tree.total_width, plot_tree.y_off)

    # 绘制特征值，并计算父节点和子节点的中心位置，添加标签信息
    plot_mid_text(ax, center_ptr, parent_ptr, node_txt)
    plot_node(ax, first_str, center_ptr, parent_ptr, decisionNode)

    second_dict = mytree
    # 采用的自顶向下的绘图，需要依次递减Y坐标
    plot_tree.y_off -= 1.0 / plot_tree.total_depth

    # 遍历子节点，如果是叶子节点，则绘制叶子节点，否则，递归调用plot_tree()
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == "dict":
            plot_tree(ax, second_dict[key], center_ptr, str(key))
        else:
            plot_tree.x_off += 1.0 / plot_tree.total_width
            plot_mid_text(ax, (plot_tree.x_off, plot_tree.y_off), center_ptr, str(key))
            plot_node(ax, second_dict[key], (plot_tree.x_off, plot_tree.y_off), center_ptr, leafNode)

    # 在绘制完所有子节点之后，需要增加Y的偏移
    plot_tree.y_off += 1.0 / plot_tree.total_depth


def create_plot(in_tree):
    fig = plt.figure(1, facecolor="white")
    fig.clf()

    ax_props = dict(xticks=[], yticks=[])
    ax = plt.subplot(111, frameon=False, **ax_props)
    plot_tree.total_width = float(get_num_leafs(in_tree))
    plot_tree.total_depth = float(get_tree_depth(in_tree))
    plot_tree.x_off = -0.5 / plot_tree.total_width
    plot_tree.y_off = 1.0
    plot_tree(ax, in_tree, (0.5, 1.0), "")
    #     plot_node(ax, "a decision node", (0.5, 0.1), (0.1, 0.5), decision_node)
    #     plot_node(ax, "a leaf node", (0.8, 0.1), (0.3, 0.8), leaf_node)
    plt.show()


def paintTree(tree):
    print()
    # create_plot(tree['8'])


def createTree(trainset, trainlabel, testset, testlabel):
    # pd.DataFrame(trainset, dtype=object)
    # pd.DataFrame(testset, dtype=object)
    dataset = [np.append(trainset[i], trainlabel[i]) for i in range(len(trainlabel))]
    labels = list(range(len(testset[0])))
    tree = createChildTree(dataset, labels)
    paintTree(tree)
    model_label = modelTree(tree, testset)
    print(accuracy(model_label, testlabel))
    print(macroF1(model_label, testlabel))
    print(microF1(model_label, testlabel))


def dataChange(trainset):
    # base = [[str(np.linalg.norm(np.array([train[0],train[1]])-np.array([train[2],train[3]])))
    #                  ,str(np.linalg.norm(np.array([train[0],train[1]])-np.array([train[4],train[5]])))
    #                  ,str(np.linalg.norm(np.array([train[2],train[3]])-np.array([train[4],train[5]])))] for train in trainset]
    base = [[str(abs(train[0] - train[2])),str(abs(train[1] - train[3])),
             str(abs(train[2] - train[4])),str(abs(train[3] - train[5])),
             str(abs(train[4] - train[0])),str(abs(train[5] - train[1]))] for train in
            trainset]
    return base

if __name__ == '__main__':
    trainset = pd.DataFrame(pd.read_csv('trainset.csv'), dtype=str)
    trainlabel = trainset['6']
    trainset = trainset.drop(['6'], axis=1)
    testset = pd.DataFrame(pd.read_csv('testset.csv'), dtype=str)
    testlabel = testset['6']
    testset = testset.drop(['6'], axis=1)
    # trainset = pd.DataFrame(pd.read_csv('trainset.csv'))
    # trainlabel = trainset['6']
    # trainset = trainset.drop(['6'], axis=1)
    # for index in ('0', '2', '4'):
    #     trainset[index] = trainset[index].replace(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'], [1, 2, 3, 4, 5, 6, 7, 8])
    # testset = pd.DataFrame(pd.read_csv('testset.csv'))
    # testlabel = testset['6']
    # testset = testset.drop(['6'], axis=1)
    # for index in ('0', '2', '4'):
    #     testset[index] = testset[index].replace(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'], [1, 2, 3, 4, 5, 6, 7, 8])
    # clf = tree.DecisionTreeClassifier(criterion='entropy')
    # clf = clf.fit(trainset, trainlabel)
    # result = clf.predict(testset)
    # print(accuracy(result, testlabel))
    # print(macroF1(result, testlabel))
    # print(microF1(result, testlabel))
    # print(testlabel.values)

    # createTree(dataChange(trainset.values), trainlabel.values, dataChange(testset.values), testlabel.values)
    createTree(trainset.values, trainlabel.values, testset.values, testlabel.values)