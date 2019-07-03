import numpy as np
import pandas as pd
import decisionTree


def linear(x, y, sigma):
    x = np.mat(x)
    return x * y.T


def guass(x, y, sigma):
    x = np.mat(x)
    row, col = np.shape(x)
    k = np.zeros((row, 1))
    for i in range(row):
        delta_row = x[i, :] - y
        k[i] = delta_row*delta_row.T
    k = np.exp(k / (-1*sigma**2))
    return k


class SMOSolver:
    def __init__(self, trainset, trainlabel, C, kernel, sigma, eps):
        self.x = np.mat(trainset)
        self.y = np.transpose(np.mat(trainlabel))
        self.C = C
        self.n = self.x.shape[0]
        self.alphas = np.mat(np.zeros((self.n, 1)))
        self.b = 0
        self.eps = eps
        self.cache = np.mat(np.zeros((self.n, 2)))
        self.k = np.mat(np.zeros((self.n, self.n)))
        for i in range(self.n):
            self.k[:, i] = kernel(self.x, self.x[i, :], sigma)
            print(i)


def calcEk(solver, k):
    fxk = float(np.multiply(solver.alphas, solver.y).T * solver.k[:, k] + solver.b)
    Ek = fxk - float(solver.y[k])
    return Ek


def fixAlpha(alpha, low, high):
    if alpha > high:
        alpha = high
    if alpha < low:
        alpha = low

    return alpha


def selectJRandom(i, m):
    j = i
    while j == i:
        j = int(np.random.uniform(0, m))
    return j


def selectJ(i, solver, ei):
    max_k = -1
    max_delta_e = 0
    ej = 0
    solver.cache[i] = [1, ei]
    valid_cache_list = np.nonzero(solver.cache[:,0].A)[0]
    if len(valid_cache_list) > 1:
        for k in valid_cache_list:
            if k == i:
                continue
            ek = calcEk(solver, k)
            delta_e = abs(ei - ek)
            if delta_e > max_delta_e:
                max_k = k
                max_delta_e = delta_e
                ej = ek
        return max_k, ej
    else:
        j =selectJRandom(i, solver.n)
        ej = calcEk(solver, j)
        return j, ej


def updateEk(solver, k):
    ek = calcEk(solver, k)
    solver.cache[k] = [1, ek]


def innerL(i, solver):
    ei = calcEk(solver, i)
    if ((solver.y[i]*ei<-solver.eps) and (solver.alphas[i]<solver.C)) or \
            ((solver.y[i]*ei>solver.eps) and (solver.alphas[i]>solver.C)):
        j, ej = selectJ(i, solver, ei)
        alpha_i_old = solver.alphas[i].copy()
        alpha_j_old = solver.alphas[j].copy()

        if solver.y[i] != solver.y[j]:
            L = max(0, solver.alphas[j] - solver.alphas[i])
            H = max(solver.C, solver.C + solver.alphas[j] - solver.alphas[i])
        else:
            L = max(0, solver.alphas[j] + solver.alphas[i] - solver.C)
            H = max(solver.C, solver.alphas[j] + solver.alphas[i])
        if H == L:
            return 0

        eta = 2.0 * solver.k[i, j] - solver.k[i, i] - solver.k[j, j]
        if eta >= 0:
            return 0
        solver.alphas[j] -= solver.y[j] * (ei - ej) / eta
        solver.alphas[j] = fixAlpha(solver.alphas[j], L, H)
        if abs(solver.alphas[j] - alpha_j_old) < 0.0001:
            return 0
        solver.alphas[i] += solver.y[j] * solver.y[i] * (alpha_j_old - solver.alphas[j])
        updateEk(solver, i)

        b1 = solver.b - ei - solver.y[i] * (solver.alphas[i] - alpha_i_old) *solver.k[i, i] - \
            solver.y[j] * (solver.alphas[j] - alpha_j_old) * solver.k[i, j]
        b2 = solver.b - ej - solver.y[i] * (solver.alphas[i] - alpha_i_old) *solver.k[i, j] - \
            solver.y[j] * (solver.alphas[j] - alpha_j_old) * solver.k[j, j]
        if 0 < solver.alphas[i] < solver.C:
            solver.b = b1
        elif 0 < solver.alphas[j] < solver.C:
            solver.b = b2
        else:
            solver.b = (b1 + b2) /2.0
        return 1
    else:
        return 0

def softSVM(trainset, trainlabel, sigma, C, max_iter, kernel, eps):
    solver = SMOSolver(trainset, trainlabel, C, kernel, sigma, eps)
    print('initial success')
    iter = 0
    entire_set = True
    alpha_pairs_changed = 0
    while iter<max_iter and (alpha_pairs_changed>0 or entire_set):
        alpha_pairs_changed = 0
        if entire_set:
            for i in range(solver.n):
                alpha_pairs_changed += innerL(i, solver)
                if alpha_pairs_changed == 1:
                    print('entire search')
                    print(alpha_pairs_changed)
            iter += 1
        else:
            non_bound_is = np.nonzero((solver.alphas.A > 0)*(solver.alphas.A < C))[0]
            for i in non_bound_is:
                alpha_pairs_changed += innerL(i, solver)
                if alpha_pairs_changed == 1:
                    print('partly search')
                    print(alpha_pairs_changed)
            iter += 1
        if entire_set:
            entire_set = False
        elif 0 == alpha_pairs_changed:
            entire_set = True

    return solver.b, solver.alphas

    # g = []
    # is_continue = False
    # maxi = 0
    # for i in range(len(solution) - 1):
    #     value = solution[0]
    #     for j in range(len(solution) - 1):
    #         value += solution[j + 1] * trainlabel[j] * kernel(trainset[i], trainset[j], sigma)
    #     g.append(value)
    #     if maxi == 0 and 0 < solution[i + 1] < C and abs(trainlabel[i] * g[i] - 1) > 0.01:
    #         maxi = i
    #         is_continue = True
    #     elif maxi == 0 and solution[i + 1] == 0 and trainlabel[i] * g[i] < 1:
    #         maxi = i
    #         is_continue = True
    #     elif maxi == 0 and solution[i + 1] == C and trainlabel[i] * g[i] > 1:
    #         maxi = i
    #         is_continue = True
    # if abs(linear(solution[1:], trainlabel, sigma)) < 0.0001 and not is_continue:
    #     return solution
    # maxvalue = 0
    # maxj = 0
    # for j in range(len(solution) - 1):
    #     if maxi != j:
    #         dis = abs(g[maxi]-g[j]+trainlabel[j]-trainlabel[maxi])
    #         if dis > maxvalue:
    #             maxj = j
    #             maxvalue = dis
    # n = kernel(trainset[maxi], trainset[maxi], sigma) + \
    #     kernel(trainset[maxj], trainset[maxj], sigma) - \
    #     2 * kernel(trainset[maxi], trainset[maxj], sigma)
    # ai_old = solution[maxi + 1]
    # aj_old = solution[maxj + 1]
    # aj_new = aj_old + trainlabel[maxj]*(g[maxi]-g[maxj]+trainlabel[maxj]-trainlabel[maxi])/n
    # L = max(0, ai_old + aj_old - C)
    # H = min(C, aj_old + ai_old)
    # if aj_new > H:
    #     aj_new = H
    # elif aj_new < L:
    #     aj_new = L
    # ai_new = ai_old + trainlabel[maxi]*trainlabel[maxj]*(aj_old-aj_new)
    # solution[maxi + 1] = ai_new
    # solution[maxj + 1] = aj_new
    # solution[0] = (-g[maxi]-g[maxj] -
    #             trainlabel[maxi] * (ai_new - ai_old) *
    #             (kernel(trainset[maxi], trainset[maxi], sigma) + kernel(trainset[maxi], trainset[maxj], sigma)) -
    #             trainlabel[maxj] * (aj_new - aj_old) *
    #             (kernel(trainset[maxj], trainset[maxj], sigma) + kernel(trainset[maxj], trainset[maxi], sigma)) +
    #             2 * solution[0]
    #             )/2
    #
    # return SMO(trainset, trainlabel, sigma, C, solution, kernel)


def multiClassSVM(trainset, trainlabel, testset, testlabel):
    kernel = guass
    sigma = 1
    C = 5
    max_iter = 100
    eps = 0
    bgroup = {}
    alphagroup = {}
    labels = {}
    for value in set(trainlabel):
        labels[value] = [1 if label == value else -1 for label in trainlabel]
        b, alphas = softSVM(trainset,
                labels[value],
                sigma, C, max_iter, kernel, eps)
        bgroup[value] = b
        alphagroup[value] = alphas
    ypred = []
    for test in testset:
        max = 0
        max_result = None
        for i in bgroup.keys():
            value = np.multiply(alphagroup[i], np.mat(labels[i]).T).T * kernel(trainset, np.mat(test), sigma) + bgroup[i]
            if value > max:
                max_result = i
                max = value
        ypred.append(max_result)
    print(ypred)
    print(decisionTree.accuracy(ypred, testlabel))
    print(decisionTree.microF1(ypred, testlabel))
    print(decisionTree.macroF1(ypred, testlabel))

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
    multiClassSVM(trainset.values[::100], trainlabel.values[::100], trainset.values, trainlabel.values)
