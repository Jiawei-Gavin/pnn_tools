from functional import *

if __name__ == '__main__':
    solver = Solver()

    # tutorial 01 -- confusion matrix
    # solver.confusion_matrix(3, 1, 1, 2)

    # tutorial 02 -- cal gx(give w, x, w0)
    # w = np.transpose([[2, 1]])
    # x = np.transpose([[3, 3]])
    # w0 = -5
    # solver.cal_gx(w, x, w0)

    # tutorial 02 -- cal cal gx(give a, x)
    a = np.transpose([[-5, 2, 1]])
    x = np.transpose([[2, 2]])
    solver.cal_gx(a, x)

    # tutorial 02 -- batch perceptron learning algorithm
    # x = np.transpose([[1, 5], [2, 5], [4, 1], [5, 1]])
    # epoch = 3
    # classx = [1, 1, -1, -1]
    # a = np.transpose([-25, 6, 3])
    # eta = 1
    # solver.batch_perceptron_learning_algorithm(epoch, x, classx, a, eta, bool(0))

    # tutorial 02 -- sequential perceptron learning algorithm
    # x = np.transpose([[1, 5], [2, 5], [4, 1], [5, 1]])
    # epoch = 3
    # classx = [1, 1, -1, -1]
    # a = np.transpose([-25, 6, 3])
    # eta = 1
    # solver.sequential_perceptron_learning_algorithm(epoch, x, classx, a, eta, bool(1))


