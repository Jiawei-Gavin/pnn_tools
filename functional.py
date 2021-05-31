from prettytable import PrettyTable
import numpy as np
import copy
import math
from sympy import *

np.set_printoptions(suppress=True)


def Heaviside(w, x):
    wx = np.dot(w, x)
    if wx > 0:
        return 1
    elif wx == 0:
        return 0.5
    else:
        return 0


def Symmetric_sigmoid(hidden_x):
    return 2 / (1 + np.exp(-2 * hidden_x)) - 1


def Logarithmic_sigmoid(output_x):
    return 1 / (1 + np.exp(-1 * output_x))


class Solver:
    def __init__(self):
        pass

    # tutorial 01 -- confusion matrix
    def confusion_matrix(self, TP, TN, FP, FN):
        error_rate = (FP + FN) / (TP + TN + FP + FN)
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        recall = TP / (TP + FN)
        precision = TP / (TP + FP)
        f1_score = 2 * TP / (2 * TP + FP + FN)
        # f1_score = (2*recall*precision)/(recall+precision)
        table = PrettyTable(["error_rate", "accuracy", "recall", "precision", "f1_score"])
        table.title = "---- confusion matrix ----"
        table.align = "c"
        table.add_row([error_rate, accuracy, recall, precision, f1_score])
        print(table)

    # tutorial 02 -- cal gx(give w, x, w0)
    def cal_gx_wxw0(self, w, x, w0):
        gx = np.dot(np.transpose(w), x) + w0
        table = PrettyTable(["gx"])
        table.title = "---- cal gx(give - w, x, w0) ----"
        table.add_row([gx[0][0]])
        print(table)

    # tutorial 02 -- cal gx(give a, x)
    def cal_gx_ax(self, a, x):
        y = np.vstack((1, x))
        gx = np.dot(np.transpose(a), y)
        table = PrettyTable(["gx"])
        table.title = "---- cal gx(give a, x) ----"
        table.add_row([gx[0][0]])
        print(table)

    # tutorial 02 -- batch perceptron learning algorithm
    def batch_perceptron_learning_algorithm(self, epoch, x, classx, a, eta, sample_normalisation):
        classx_ = np.ones(np.size(x, 1))
        y = np.vstack((classx_, x))
        table = PrettyTable(["epoch", "gx", "misclassified", "a"])
        table.title = "---- batch perceptron learning algorithm ----"
        table.align = "c"
        if sample_normalisation:
            for i in range(len(classx)):
                if classx[i] < 0:
                    y[:, i] = -(y[:, i])
            for epochi in range(epoch):
                gx = np.dot(np.transpose(a), y)
                misclassified = np.zeros(np.size([gx]))
                for i in range(np.size([gx], 1)):
                    if gx[i] <= 0:
                        misclassified[i] = 1
                for i in range(np.size([gx], 1)):
                    if misclassified[i] == 1:
                        a = a + eta * y[:, i]
                table.add_row([epochi + 1, gx, misclassified, a])
        else:
            for epochi in range(epoch):
                gx = np.dot(np.transpose(a), y)
                misclassified = np.zeros(np.size([gx]))
                for i in range(np.size([gx], 1)):
                    if gx[i] > 0:
                        classgx = 1
                    else:
                        classgx = -1
                    if classgx != classx[i]:
                        misclassified[i] = 1
                for i in range(np.size([gx], 1)):
                    if misclassified[i] == 1:
                        a = a + eta * classx[i] * y[:, i]
                table.add_row([epochi + 1, gx, misclassified, a])
        print(table)

    # tutorial 02 -- sequential perceptron learning algorithm
    def sequential_perceptron_learning_algorithm(self, epoch, x, classx, a, eta, sample_normalisation):
        classx_ = np.ones(np.size(x, 1))
        y = np.vstack((classx_, x))
        table = PrettyTable(["iteration", "gx", "a"])
        table.title = "---- sequential perceptron learning algorithm ----"
        table.align = "c"
        iteration = 1
        if sample_normalisation:
            for i in range(len(classx)):
                if classx[i] < 0:
                    y[:, i] = -(y[:, i])
            for epochi in range(epoch):
                for i in range(np.size(y, 1)):
                    gx = np.dot(np.transpose(a), y[:, i])
                    if gx <= 0:
                        a = a + eta * y[:, i]
                    table.add_row([iteration, gx, a])
                    iteration = iteration + 1
        else:
            for epochi in range(epoch):
                for i in range(np.size(y, 1)):
                    gx = np.dot(np.transpose(a), y[:, i])
                    if gx > 0:
                        classgx = 1
                    else:
                        classgx = -1
                    if classgx != classx[i]:
                        a = a + eta * classx[i] * y[:, i]
                    table.add_row([iteration, gx, a])
                    iteration = iteration + 1
        print(table)

    # tutorial 02 -- sequential multiclass perceptron learning algorithm
    def sequential_multiclass_perceptron_learning_algorithm(self, epoch, x, classx, a, eta):
        classx_ = np.ones(np.size(x, 1))
        y = np.vstack((classx_, x))
        table = PrettyTable(["iteration", "y", "gx", "gxclass", "class", "a1", "a2", "a3"])
        table.title = "--- sequential multiclass perceptron learning algorithm ---"
        table.align = "c"
        result = []
        iteration = 1
        for epochi in range(epoch):
            for i in range(np.size(y, 1)):
                gx = np.zeros(np.size(a, 1))
                for ci in range(np.size(a, 1)):
                    gx[ci] = np.dot(np.transpose(a[:, ci]), y[:, i])
                gx_index = [j[0] for j in sorted(enumerate(gx), key=lambda k: k[1])]
                gxclass = gx_index[-1] + 1
                w = classx[i]
                if gxclass != w:
                    a[:, w - 1] = a[:, w - 1] + eta * y[:, i]
                    a[:, gxclass - 1] = a[:, gxclass - 1] - eta * y[:, i]
                a1 = copy.deepcopy(a[:, 0])
                a2 = copy.deepcopy(a[:, 1])
                a3 = copy.deepcopy(a[:, 2])
                result.append((iteration, y[:, i], gx, gxclass, w, a1, a2, a3))
                iteration = iteration + 1
        for row in result:
            table.add_row(row)
        print(table)

    # tutorial 02 -- sequential WidrowHoff learning algorithm
    def sequential_WidrowHoff_learning_algorithm(self, epoch, x, classx, a, eta, b):
        classx_ = np.ones(np.size(x, 1))
        y = np.vstack((classx_, x))
        for i in range(len(classx)):
            if classx[i] < 0:
                y[:, i] = -(y[:, i])
        table = PrettyTable(('iteration', 'a', 'y', 'ay', 'a_new'))
        table.title = "--- sequential WidrowHoff learning algorithm ---"
        table.align = "c"
        result = []
        iteration = 1
        for epochi in range(epoch):
            for i in range(np.size(y, 1)):
                a_prev = a
                ay = np.dot(a, y[:, i])
                a = a + eta * (b[i] - ay) * y[:, i]
                result.append((iteration, np.round(a_prev, 4), np.round(y[:, i], 4), np.round(ay, 4), np.round(a, 4)))
                iteration = iteration + 1
        for row in result:
            table.add_row(row)
        print(table)

    # tutorial 03 -- sequential Delta learning rule
    def sequential_Delta_learning_rule(self, epoch, w, x, t, eta):
        x_ = np.ones(np.size(x, 1))
        x = np.vstack((x_, x))

        table = PrettyTable(('iteration', 'x', 't', 'y=H(wx)', 't-y', "delta", "w"))
        table.title = "--- sequential Delta learning rule ---"
        table.align = "c"
        iteration = 1
        for epochi in range(epoch):
            for i in range(len(x_)):
                y = Heaviside(w, x[:, i])
                t_y = t[i] - y
                delta = eta * t_y * np.transpose(x[:, i])
                w = w + delta
                table.add_row([iteration, x[:, i], t[i], y, t_y, delta, w])
                iteration = iteration + 1
        print(table)

    # tutorial 03 -- batch Delta learning rule
    def batch_Delta_learning_rule(self, epoch, w, x, t, eta):
        x_row = np.size(x, 0)
        x_col = np.size(x, 1)
        x_ = np.ones(x_col)
        x = np.vstack((x_, x))
        table = PrettyTable(('iteration', 'x', 't', 'y=H(wx)', 't-y', "delta"))
        table.title = "--- batch Delta learning rule ---"
        table.align = "c"
        iteration = 1
        for epochi in range(epoch):
            delta = np.zeros((x_col, x_row + 1))
            for i in range(len(x_)):
                y = Heaviside(w, x[:, i])
                t_y = t[i] - y
                delta[i, :] = eta * t_y * np.transpose(x[:, i])
                table.add_row([iteration, x[:, i], t[i], y, t_y, delta[i, :]])

                iteration = iteration + 1
            sum_delta = sum(delta[:])
            w = w + sum_delta
            table.add_row(["sum_delta", sum_delta, "w", w, "", ""])
        print(table)

    # tutorial 04 -- neural network
    def neural_network(self, x, wji, wj0, wkj, wk0, function1="Symmetric_sigmoid", function2="Logarithmic_sigmoid"):
        input_x = x
        hidden_x = np.dot(wji, input_x) + wj0
        if function1 == "Symmetric_sigmoid":
            hidden_y = Symmetric_sigmoid(hidden_x)
        else:
            hidden_y = Logarithmic_sigmoid(hidden_x)
        y = hidden_y
        output_x = np.dot(wkj, hidden_y) + wk0
        if function2 == "Logarithmic_sigmoid":
            output_z = Logarithmic_sigmoid(output_x)
        else:
            output_z = Symmetric_sigmoid(output_x)
        table = PrettyTable(('pattern', 'y', 'z'))
        table.title = "--- neural network ---"
        table.align = "c"
        for i in range(np.size(output_z, 1)):
            table.add_row([i + 1, np.round(y[:, i], 4), np.round(output_z[:, i], 4)])
        print(table)

    # tutorial 04 -- RBF neural network -- give x,c,t -- compute w
    def RBF_neural_network_w(self, x, c, t):
        pmax = np.linalg.norm(c)
        yx = np.ones((np.size(x, 1), np.size(x, 0) + 1))
        for i in range(np.size(x, 1)):
            yx[:, 0][i] = math.exp(
                -np.power(np.linalg.norm(x[:, i] - c[0]), 2) / (2 * math.pow((pmax / math.sqrt(2 * 2)), 2)))
            yx[:, 1][i] = math.exp(
                -np.power(np.linalg.norm(x[:, i] - c[1]), 2) / (2 * math.pow((pmax / math.sqrt(2 * 2)), 2)))
        w = np.dot(np.linalg.inv(np.dot(np.transpose(yx), yx)), np.dot(np.transpose(yx), t))
        table = PrettyTable(('yx', 'w'))
        table.title = "--- RBF neural network -- give x,c,t -- compute w ---"
        table.align = "c"
        table.add_row([yx, w])
        print(table)

    # tutorial 04 -- RBF neural network -- give x,c,w -- compute class
    def RBF_neural_network_class(self, x, c, w):
        pmax = np.linalg.norm(c)
        yx = np.ones((np.size(x, 1), np.size(x, 0) + 1))
        for i in range(np.size(x, 1)):
            yx[:, 0][i] = math.exp(
                -np.power(np.linalg.norm(x[:, i] - c[0]), 2) / (2 * math.pow((pmax / math.sqrt(2 * 2)), 2)))
            yx[:, 1][i] = math.exp(
                -np.power(np.linalg.norm(x[:, i] - c[1]), 2) / (2 * math.pow((pmax / math.sqrt(2 * 2)), 2)))
        z = np.dot(yx, w)
        clazz = np.zeros((np.size(z), 1))
        for i in range(np.size(z)):
            if z[i] > 0.5:
                clazz[i] = 1
            else:
                clazz[i] = 0
        table = PrettyTable(('yx', 'z', 'class'))
        table.title = "--- RBF neural network -- give x,c,w -- compute class ---"
        table.align = "c"
        table.add_row([yx, z, clazz])
        print(table)

    # tutorial 06 -- GAN

    # tutorial 07 -- Karhunen Loeve Transform
    def Karhunen_Loeve_Transform(self, x, num):
        u = np.mean(np.mat(x), axis=1)
        x_ = x - u
        covariance = np.cov(x_, bias=True)
        [E, V] = np.linalg.eigh(covariance)
        E = np.mat(np.diag(E))
        for i in range(np.size(V, 1) - num):
            V = np.delete(V, 0, axis=1)
            E = np.delete(E, 0, axis=1)
        y = np.dot(np.transpose(V), x_)
        table = PrettyTable(('y'))
        table.title = "--- Karhunen Loeve Transform ---"
        table.align = "c"
        table.add_row([y])
        print(table)

    # tutorial 07 -- batch Ojas Learning rule
    def batch_Ojas_Learning_rule(self, x, w, eta, epoch):
        u = np.mean(np.mat(x), axis=1)
        x_ = x - u
        table = PrettyTable(('iteration', 'x', 'y', 'x-yw', 'ηy(x-yw)', "w"))
        table.title = "--- batch Ojas Learning rule ---"
        table.align = "c"
        iteration = 1
        for epochi in range(epoch):
            eta_y_xyw = np.zeros((np.size(x_, 1), np.size(x_, 0)))
            for i in range(np.size(x_, 1)):
                y = w * x_[:, i]
                x_yw = np.transpose(x_[:, i]) - y * w
                eta_y_xyw[i, :] = eta * y * x_yw
                table.add_row([iteration, np.transpose(x_[:, i]), y, x_yw, eta_y_xyw[i, :], ""])
                iteration = iteration + 1
            total_weight_change = sum(eta_y_xyw[:])
            w = w + total_weight_change
            table.add_row(["total weight change:", "", "", "", total_weight_change, w])
        print(table)

    # tutorial 07 -- Fishers method -- LDA
    def Fishers_method(self, x, classx, wt):
        m = np.zeros((np.size(x, 0), np.size(np.unique(classx))))
        for i in range(np.size(np.unique(classx))):
            num = 0
            tmp = np.mat([[0, 0]])
            for j in range(np.size(classx)):
                if np.unique(classx)[i] == classx[j]:
                    num = num + 1
                    tmp = tmp + x[:, j]
            m[:, i] = tmp / num
        sb = np.power((np.dot(wt, (m[:, 0] - m[:, 1]))), 2)
        sw = 0
        for j in range(np.size(classx)):
            if 1 == classx[j]:
                sw = sw + np.power(np.dot(wt, (x[:, j] - m[:, 0])), 2)
            elif 2 == classx[j]:
                sw = sw + np.power(np.dot(wt, (x[:, j] - m[:, 1])), 2)
        Jw = sb / sw
        table = PrettyTable(('m1', "m2", 'sb', 'sw', 'Jw'))
        table.title = "--- Fishers method -- LDA ---"
        table.align = "c"
        table.add_row([m[:, 0], m[:, 1], sb, sw, Jw])
        print(table)

    # tutorial 07 -- Extreme Learning Machine
    def Extreme_Learning_Machine(self, V, w, x):
        x_ = np.ones(np.size(x, 1))
        X = np.vstack((x_, x))
        VX = np.dot(V, X)
        Y = copy.copy(VX)
        for i in range(np.size(Y, 0)):
            for j in range(np.size(Y, 1)):
                if Y[i][j] > 0:
                    Y[i][j] = 1
                else:
                    Y[i][j] = 0
        Z = np.dot(w, np.vstack((x_, Y)))
        table = PrettyTable(('X', "VX", 'Y', 'Z'))
        table.title = "--- Extreme Learning Machine ---"
        table.align = "c"
        table.add_row([X, VX, Y, Z])
        print(table)

    # tutorial 07 -- best sparse code
    def best_sparse_code(self, Vt, x, yt):
        y = np.transpose(yt)
        Vt_y = np.dot(Vt, y)
        x_Vt_y = x - np.transpose(np.mat(Vt_y))
        error = np.linalg.norm(x_Vt_y)
        table = PrettyTable(('x-Vt_y', "error"))
        table.title = "--- best sparse code ---"
        table.align = "c"
        table.add_row([x_Vt_y, error])
        print(table)

    # tutorial 08 -- SVM
    def SVM(self, x, y):
        lmb1, lmb2, lmb3, lmb4, lmb5, lmb6, w0 = symbols('lmb1 lmb2 lmb3 lmb4 lmb5 lmb6 w0')
        x1, x2, x3, x4 = symbols('x1 x2 x3 x4')
        lmb = [lmb1, lmb2, lmb3, lmb4, lmb5, lmb6]
        lmb = lmb[0:np.size(np.mat(y), 1)]
        x_ = [x1, x2, x3, x4]
        sym_x = np.transpose(x_[0:np.size(x, 0)])
        w = np.dot(np.mat(x), np.transpose(np.mat(np.multiply(lmb, y))))
        lmb_y = np.sum(np.dot(lmb, y))
        y_wxw0 = np.multiply(y, (np.dot(np.transpose(w), x) + w0))
        one = np.ones((np.size(y_wxw0, 0), np.size(y_wxw0, 1)))
        y_wxw0_1 = y_wxw0 - one
        result = []
        for i in range(np.size(y_wxw0_1, 1)):
            result.append(np.diag(y_wxw0_1[:, i])[0])
        result.append(lmb_y)
        result_ = []
        for i in range(np.size(lmb)):
            result_.append(lmb[i])
        result_.append(w0)
        solution = solve(result, result_)
        hyperplane = np.dot(np.transpose(w), sym_x) + w0
        hyperplane = np.diag(hyperplane)[0]
        hyperplane = re.subs(hyperplane, solution)
        table = PrettyTable(('lmbda', 'w方程', 'y(wx+w0)-1方程', 'Σlmbda*y', '解方程结果', 'hyperplane'))
        table.title = "--- SVM ---"
        table.align = "c"
        table.add_row([lmb, w, np.transpose(y_wxw0_1), lmb_y, solution, str(hyperplane) + ' == 0'])
        print(table)

    # tutorial 09 -- ENSEMBLE -- AdaBoost algorithm
    def adaBoost_algorithm(self, x, epoch, h):
        h1x, h2x, h3x, h4x, h5x, h6x, h7x, h8x = symbols('h1x, h2x, h3x, h4x, h5x, h6x, h7x, h8x ')
        hx = [h1x, h2x, h3x, h4x, h5x, h6x, h7x, h8x]
        classifier = 0
        w = 1 / np.size(x, 1) * np.ones((1, np.size(x, 1)))
        table = PrettyTable(('epoch', "train_error", '本次选择的hx', 'ε', 'α', '未归一化权重', '归一化权重', 'classifier分类器'))
        table.title = "--- ENSEMBLE -- AdaBoost algorithm ---"
        table.align = "l"
        for epochi in range(epoch):
            tmp = copy.deepcopy(h)
            train_error = np.zeros((1, np.size(np.mat(hx), 1)))
            for i in range(np.size(np.mat(hx), 1)):
                for j in range(len(h[i])):
                    if tmp[i][j] < 0:
                        tmp[i][j] = 1
                    else:
                        tmp[i][j] = 0
                train_error[:, i] = np.dot(tmp[i], np.transpose(w))
            train_error = train_error[0]
            sort_index = [j[0] for j in sorted(enumerate(train_error), key=lambda k: k[1])]
            sort_error = sorted(train_error)
            index = sort_index[0]
            if sort_error[0] > 0.5:
                print("NO WAY!!!!!")
                break
            e1 = train_error[index]
            a = 1 / 2 * log((1 - e1) / e1)
            a = round(a, 4)
            a_y_h = np.dot(a, np.mat(h)[index, :])
            w_e_ayh = np.multiply(w, np.power(math.e, np.dot(-1, a_y_h)))
            z = np.sum(w_e_ayh)
            w_new = w_e_ayh / z
            w = w_new
            classifier = classifier + np.dot(a, hx[index])
            table.add_row([epochi + 1, train_error, 'h' + str(index + 1) + 'x', e1, a, w_e_ayh, w_new, classifier])
        print(table)

    # tutorial 10 -- K-means algorithm
    def Kmeans_algorithm(self, S, m1, m2):
        Iteration = 1
        table = PrettyTable(('Iteration', 'x', '||x-m1||', '||x-m2||', 'class', 'new_m1', 'new_m2'))
        table.title = "--- K-means algorithm ---"
        while Iteration != 0:
            result = []
            clazz1 = []
            clazz2 = []
            for i in range(np.size(S, 1)):
                x_m1 = np.linalg.norm(S[:, i] - m1)
                x_m2 = np.linalg.norm(S[:, i] - m2)
                if x_m1 < x_m2:
                    clazz = 1
                    clazz1.append(i)
                else:
                    clazz = 2
                    clazz2.append(i)
                result.append(('', S[:, i], x_m1, x_m2, clazz, '', ''))
            old_m1 = copy.deepcopy(m1)
            old_m2 = copy.deepcopy(m2)
            m1 = 0
            m2 = 0
            for i in range(len(clazz1)):
                m1 += S[:, clazz1[i]]
                m2 += S[:, clazz2[i]]
            m1 = m1 / len(clazz1)
            m2 = m2 / len(clazz2)

            result[0] = np.append([Iteration], np.array(result[0][1:], dtype=object))
            for row in result:
                table.add_row(row)
            table.add_row(['', '', '', '', '', m1, m2])
            if (old_m1 == m1).all() and (old_m2 == m2).all():
                break
            Iteration += 1
        print(table)

    # tutorial 10 -- competitive learning algorithm (without normalisation)
    def competitive_learning_algorithm(self, S, m1, m2, m3, eta, epoch, x, test_x=None):
        table = PrettyTable(('Iteration', 'x', '||x-m1||', '||x-m2||', '||x-m3||', 'j', 'mj'))
        table.title = "--- competitive learning algorithm (without normalisation) ---"
        result = []
        for epochi in range(epoch):
            for i in range(np.size(x, 1)):
                x_m1 = np.linalg.norm(x[:, i] - m1)
                x_m2 = np.linalg.norm(x[:, i] - m2)
                x_m3 = np.linalg.norm(x[:, i] - m3)
                if x_m1 == min(x_m1, x_m2, x_m3):
                    j = 1
                    m1 = m1 + eta * (x[:, i] - m1)
                    result.append((i, x[:, i], x_m1, x_m2, x_m3, j, 'm' + str(j) + str(m1)))
                elif x_m2 == min(x_m1, x_m2, x_m3):
                    j = 2
                    m2 = m2 + eta * (x[:, i] - m2)
                    result.append((i, x[:, i], x_m1, x_m2, x_m3, j, 'm' + str(j) + str(m2)))
                elif x_m3 == min(x_m1, x_m2, x_m3):
                    j = 3
                    m3 = m3 + eta * (x[:, i] - m3)
                    result.append((i + 1, x[:, i], x_m1, x_m2, x_m3, j, 'm' + str(j) + str(m3)))
        for row in result:
            table.add_row(row)
        table.add_row(['m1:', m1, 'm2:', m2, 'm3:', m3, ''])
        print(table)
        clazz = []
        for i in range(np.size(S, 1)):
            x_m1 = np.linalg.norm(S[:, i] - m1)
            x_m2 = np.linalg.norm(S[:, i] - m2)
            x_m3 = np.linalg.norm(S[:, i] - m3)
            if x_m1 == min(x_m1, x_m2, x_m3):
                clazz.append((S[:, i], 1))
            elif x_m2 == min(x_m1, x_m2, x_m3):
                clazz.append((S[:, i], 2))
            elif x_m3 == min(x_m1, x_m2, x_m3):
                clazz.append((S[:, i], 3))
        table = PrettyTable(('x', 'class'))
        for row in clazz:
            table.add_row(row)
        print(table)
        if test_x is not None:
            test_x_m1 = np.linalg.norm(test_x - m1)
            test_x_m2 = np.linalg.norm(test_x - m2)
            test_x_m3 = np.linalg.norm(test_x - m3)
            clazz = 0
            if test_x_m1 == min(test_x_m1, test_x_m2, test_x_m3):
                clazz = 1
            elif test_x_m2 == min(test_x_m1, test_x_m2, test_x_m3):
                clazz = 2
            elif test_x_m3 == min(test_x_m1, test_x_m2, test_x_m3):
                clazz = 3
            table = PrettyTable(('test_x', '||test_x-m1||', '||test_x-m2||', '||test_x-m3||', 'class'))
            table.add_row([test_x, test_x_m1, test_x_m2, test_x_m3, clazz])
            print(table)
