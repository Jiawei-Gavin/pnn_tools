from functional import *

if __name__ == '__main__':
    solver = Solver()

    # tutorial 01 -- confusion matrix
    # solver.confusion_matrix(3, 1, 1, 2)

    # tutorial 02 -- cal gx(give w, x, w0)
    # w = np.transpose([[2, 1]])
    # x = np.transpose([[1, 1]])
    # w0 = -5
    # solver.cal_gx_wxw0(w, x, w0)

    # tutorial 02 -- cal cal gx(give a, x)
    # a = np.transpose([[-5, 2, 1]])
    # x = np.transpose([[1, 1]])
    # solver.cal_gx_ax(a, x)

    # tutorial 02 -- batch perceptron learning algorithm
    # x = np.transpose([[1, 5], [2, 5], [4, 1], [5, 1]])
    # epoch = 3
    # classx = [1, 1, -1, -1]
    # a = np.transpose([-25, 6, 3])
    # eta = 1
    # solver.batch_perceptron_learning_algorithm(epoch, x, classx, a, eta, bool(0))

    # tutorial 02 -- sequential perceptron learning algorithm
    # x = np.transpose([[1, 5], [2, 5], [4, 1], [5, 1]])
    # epoch = 2
    # classx = [1, 1, -1, -1]
    # a = np.transpose([-25, 6, 3])
    # eta = 1
    # solver.sequential_perceptron_learning_algorithm(epoch, x, classx, a, eta, bool(1))
    # ---
    # x = np.transpose([[0, 2], [1, 2], [2, 1], [-3, 1], [-2, -1], [-3, -2]])
    # epoch = 2
    # classx = [1, 1, 1, -1, -1, -1]
    # a = np.transpose([1, 0, 0])
    # eta = 1
    # solver.sequential_perceptron_learning_algorithm(epoch, x, classx, a, eta, bool(0))

    # tutorial 02 -- sequential multiclass perceptron learning algorithm
    # x = np.transpose([[1, 1], [2, 0], [0, 2], [-1, 1], [-1, -1]])
    # epoch = 3
    # classx = [1, 1, 2, 2, 3]
    # a = np.transpose([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    # eta = 1
    # solver.sequential_multiclass_perceptron_learning_algorithm(epoch, x, classx, a, eta)
