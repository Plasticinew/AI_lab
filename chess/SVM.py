import numpy as np
import pandas as pd


def linear(x, y, sigma):
    return np.dot(x, y)


def guass(x, y, sigma):
    return np.exp(-np.linalg.norm(x - y)**2/(2*sigma**2))


def SMO(trainset, trainlabel, sigma, C, solution, kernel):
    g = []
    is_continue = False
    maxi = 0
    for i in range(len(solution) - 1):
        value = solution[0]
        for j in range(len(solution) - 1):
            value += solution[j + 1] * trainlabel[j] * kernel(trainset[i], trainset[j], sigma)
        g.append(value)
        if maxi == 0 and 0 < solution[i + 1] < C and abs(trainlabel[i] * g[i] - 1) > 0.01:
            maxi = i
            is_continue = True
        elif maxi == 0 and solution[i + 1] == 0 and trainlabel[i] * g[i] < 1:
            maxi = i
            is_continue = True
        elif maxi == 0 and solution[i + 1] == C and trainlabel[i] * g[i] > 1:
            maxi = i
            is_continue = True
    if abs(linear(solution[1:], trainlabel, sigma)) < 0.0001 and not is_continue:
        return solution
    maxvalue = 0
    maxj = 0
    for j in range(len(solution) - 1):
        if maxi != j:
            dis = abs(g[maxi]-g[j]+trainlabel[j]-trainlabel[maxi])
            if dis > maxvalue:
                maxj = j
                maxvalue = dis
    n = kernel(trainset[maxi], trainset[maxi], sigma) + \
        kernel(trainset[maxj], trainset[maxj], sigma) - \
        2 * kernel(trainset[maxi], trainset[maxj], sigma)
    ai_old = solution[maxi + 1]
    aj_old = solution[maxj + 1]
    aj_new = aj_old + trainlabel[maxj]*(g[maxi]-g[maxj]+trainlabel[maxj]-trainlabel[maxi])/n
    L = max(0, ai_old + aj_old - C)
    H = min(C, aj_old + ai_old)
    if aj_new > H:
        aj_new = H
    elif aj_new < L:
        aj_new = L
    ai_new = ai_old + trainlabel[maxi]*trainlabel[maxj]*(aj_old-aj_new)
    solution[maxi + 1] = ai_new
    solution[maxj + 1] = aj_new
    solution[0] = (-g[maxi]-g[maxj] -
                trainlabel[maxi] * (ai_new - ai_old) *
                (kernel(trainset[maxi], trainset[maxi], sigma) + kernel(trainset[maxi], trainset[maxj], sigma)) -
                trainlabel[maxj] * (aj_new - aj_old) *
                (kernel(trainset[maxj], trainset[maxj], sigma) + kernel(trainset[maxj], trainset[maxi], sigma)) +
                2 * solution[0]
                )/2

    return SMO(trainset, trainlabel, sigma, C, solution, kernel)

def softSVM(trainset, trainlabel, sigma, C):
    kernel = linear if sigma == 0 else guass
    dim = len(trainlabel) + 1
    solution = np.zeros(dim)
    return SMO(trainset, trainlabel, sigma, C, solution, kernel)



def multiClassSVM(trainset, trainlabel, testset, testlabel):
    for value in set(trainlabel):
        print(softSVM(trainset,
                [1 if label == value else -1 for label in trainlabel ],
                0,0.2))

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
    multiClassSVM(trainset.values[1::200], trainlabel.values[1::200], trainset.values[1::200], trainlabel.values[1::200])
