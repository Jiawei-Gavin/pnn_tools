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
