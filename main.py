from functional import *

if __name__ == '__main__':
    solver = Solver()

    # tutorial 01 -- confusion matrix
    # solver.confusion_matrix(3, 1, 1, 2)

    # tutorial 02 -- batch perceptron learning algorithm
    # x = np.transpose([[1, 5], [2, 5], [4, 1], [5, 1]])
    # epoch = 3
    # classx = [1, 1, -1, -1]
    # a = np.transpose([-25, 6, 3])
    # eta = 1
    # solver.batch_perceptron_learning_algorithm(epoch, x, classx, a, eta, bool(1))

    # tutorial 02 -- sequential perceptron learning algorithm
    x = np.transpose([[1, 5], [2, 5], [4, 1], [5, 1]])
    epoch = 3
    classx = [1, 1, -1, -1]
    a = np.transpose([-25, 6, 3])
    eta = 1
    solver.sequential_perceptron_learning_algorithm(epoch, x, classx, a, eta, bool(1))

