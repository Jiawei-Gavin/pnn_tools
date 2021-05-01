from prettytable import PrettyTable
import numpy as np


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
        table = PrettyTable(["metrics", "result"])
        table.title = "---- confusion matrix ----"
        table.align = "l"
        table.add_row(["error_rate", error_rate])
        table.add_row(["accuracy", accuracy])
        table.add_row(["recall", recall])
        table.add_row(["precision", precision])
        table.add_row(["f1_score", f1_score])
        print(table)

    # tutorial 02 -- batch perceptron learning algorithm
    def batch_perceptron_learning_algorithm(self, epoch, x, classx, a, eta, sample_normalisation):
        classx_ = np.ones(np.size(x, 1))
        y = np.vstack((classx_, x))
        if sample_normalisation:
            for i in range(len(classx_)):
                if classx[i] < 0:
                    y[:, i] = -(y[:, i])
            for epochi in range(epoch):
                table = PrettyTable(["epoch=" + repr(epochi), "result"])
                table.title = "---- batch perceptron learning algorithm ----"
                table.align = "l"
                gx = np.dot(np.transpose(a), y)
                misclassified = np.zeros(np.size(gx))
                for i in range(np.size(gx)):
                    if gx[i] <= 0:
                        misclassified[i] = 1
                for i in range(np.size(gx)):
                    if misclassified[i] == 1:
                        a = a + eta * y[:, i]
                table.add_row(["gx", gx])
                table.add_row(["misclassified", misclassified])
                table.add_row(["a", a])
                print(table)
        else:
            for epochi in range(epoch):
                table = PrettyTable(["epoch=" + repr(epochi), "result"])
                table.title = "---- batch perceptron learning algorithm ----"
                table.align = "l"
                gx = np.dot(np.transpose(a), y)
                misclassified = np.zeros(np.size(gx))
                for i in range(np.size(gx)):
                    if gx[i] > 0:
                        classgx = 1
                    else:
                        classgx = -1
                    if classgx != classx[i]:
                        misclassified[i] = 1
                for i in range(np.size(gx)):
                    if misclassified[i] == 1:
                        a = a + eta * classx[i] * y[:, i]
                table.add_row(["gx", gx])
                table.add_row(["misclassified", misclassified])
                table.add_row(["a", a])
                print(table)
