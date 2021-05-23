from prettytable import PrettyTable
import numpy as np
import copy
import math

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

