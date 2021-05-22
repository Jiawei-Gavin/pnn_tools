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

    # tutorial 02 -- sequential WidrowHoff learning algorithm
    # x = np.transpose([[0, 2], [1, 2], [2, 1], [-3, 1], [-2, -1], [-3, -2]])
    # epoch = 2
    # classx = [1, 1, 1, -1, -1, -1]
    # a = np.transpose([1, 0, 0])
    # eta = 0.1
    # b = [1, 1, 1, 1, 1, 1]
    # solver.sequential_WidrowHoff_learning_algorithm(epoch, x, classx, a, eta, b)

    # tutorial 03 -- sequential Delta learning rule
    # x = np.transpose([[0], [1]])
    # w = [-1.5, 2]
    # epoch = 6
    # eta = 1
    # t = [1, 0]
    # solver.sequential_Delta_learning_rule(epoch, w, x, t, eta)

    # x = np.transpose([[0, 0], [0, 1], [1, 0], [1, 1]])
    # w = [0.5, 1, 1]
    # epoch = 5
    # eta = 1
    # t = [0, 0, 0, 1]
    # solver.sequential_Delta_learning_rule(epoch, w, x, t, eta)

    # tutorial 03 -- batch Delta learning rule
    # x = np.transpose([[0], [1]])
    # w = [-1.5, 2]
    # epoch = 6
    # eta = 1
    # t = [1, 0]
    # solver.batch_Delta_learning_rule(epoch, w, x, t, eta)

    # tutorial 04 -- neural network
    # x = np.transpose([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]])
    # wji = [[-0.7057, 1.9061, 2.6605, -1.1359], [0.4900, 1.9324, -0.4269, -5.1570], [0.9438, -5.4160, -0.3431, -0.2931]]
    # wj0 = [[4.8432], [0.3973], [2.1761]]
    # wkj = [[-1.1444, 0.3115, -9.9812], [0.0106, 11.5477, 2.6479]]
    # wk0 = [[2.5230], [2.6463]]
    # solver.neural_network(x, wji, wj0, wkj, wk0)

    # x = np.transpose([[0.1, 0.9]])
    # wji = [[0.5, 0], [0.3, -0.7]]
    # wj0 = [[0.2], [0]]
    # wkj = [[0.8, 1.6]]
    # wk0 = [[-0.4]]
    # solver.neural_network(x, wji, wj0, wkj, wk0, "Symmetric_sigmoid", "Symmetric_sigmoid")

    # tutorial 04 -- RBF neural network -- give x,c,t -- compute w
    # x = np.transpose([[0, 0], [0, 1], [1, 0], [1, 1]])
    # c = [[0, 0], [1, 1]]
    # t = np.transpose([[0, 1, 1, 0]])
    # solver.RBF_neural_network_w(x, c, t)


    # tutorial 04 -- RBF neural network -- give x,c,w -- compute class
    # x = np.transpose([[0.5, -0.1], [-0.2, 1.2], [0.8, 0.3], [1.8, 0.6]])
    # c = [[0, 0], [1, 1]]
    # w = np.transpose([[-2.5027, -2.5027, 2.8413]])
    # solver.RBF_neural_network_class(x, c, w)
