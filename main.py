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

    # x = np.transpose([[5, 1], [5, -1], [7, 0], [3, 0], [2, 1], [1, -1]])
    # epoch = 2
    # classx = [1, 1, 1, -1, -1, -1]
    # a = np.transpose([-25, 5, 2])
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
    #
    # x = np.transpose([[0, 2], [2, 1], [-3, 1], [-2, -1], [0, -1]])
    # w = [2, 0.5, 1]
    # epoch = 5
    # eta = 1
    # t = [1, 1, 0, 0, 0]
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
    #
    # x = np.transpose([[0.1, 0.9]])
    # wji = [[0.5, 0], [0.3, -0.7]]
    # wj0 = [[0.2], [0]]
    # wkj = [[0.8, 1.6]]
    # wk0 = [[-0.4]]
    # solver.neural_network(x, wji, wj0, wkj, wk0, "Symmetric_sigmoid", "Symmetric_sigmoid")
    # x = np.transpose([[2, -6]])
    # wji = np.transpose([[1, 0.5], [0, -3]])
    # wj0 = [[0], [-2]]
    # wkj = [[6, 7]]
    # wk0 = [[8]]
    # solver.neural_network(x, wji, wj0, wkj, wk0, "Logarithmic_sigmoid", "Logarithmic_sigmoid")

    # tutorial 04 -- RBF neural network -- give x,c,t -- compute w
    x = np.transpose([[0, 0], [0, 1], [1, 0], [1, 1]])
    c = [[0, 0], [1, 1]]
    t = np.transpose([[0, 1, 1, 0]])
    solver.RBF_neural_network_w(x, c, t)
    #
    # x = np.transpose([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    # c = [[-1, -1], [1, 1]]
    # t = np.transpose([[-1, 1, 1, -1]])
    # solver.RBF_neural_network_w(x, c, t)

    # tutorial 04 -- RBF neural network -- give x,c,w -- compute class
    # x = np.transpose([[0.5, -0.1], [-0.2, 1.2], [0.8, 0.3], [1.8, 0.6]])
    # c = [[0, 0], [1, 1]]
    # w = np.transpose([[-2.5027, -2.5027, 2.8413]])
    # solver.RBF_neural_network_class(x, c, w)

    # tutorial 07 -- Karhunen Loeve Transform
    # x = np.transpose([[1, 2, 1], [2, 3, 1], [3, 5, 1], [2, 2, 1]])
    # num = 2
    # x = np.transpose([[0, 1], [3, 5], [5, 4], [5, 6], [8, 7], [9, 7]])
    # num = 1
    # solver.Karhunen_Loeve_Transform(x, num)

    # tutorial 07 -- batch Ojas Learning rule
    # x = np.transpose([[0, 1], [3, 5], [5, 4], [5, 6], [8, 7], [9, 7]])
    # w = [-1, 0]
    # eta = 0.01
    # epoch = 6
    # solver.batch_Ojas_Learning_rule(x, w, eta, epoch)

    # tutorial 07 -- Fishers method -- LDA
    # x = np.transpose([[1, 2], [2, 1], [3, 3], [6, 5], [7, 8]])
    # classx = [1, 1, 1, 2, 2]
    # wt = [-1, 5]
    # solver.Fishers_method(x, classx, wt)
    # wt = [2, -6]
    # solver.Fishers_method(x, classx, wt)

    # tutorial 07 -- Extreme Learning Machine
    # x = np.transpose([[0, 0], [0, 1], [1, 0], [1, 1]])
    # V = [[-0.62, 0.44, -0.91], [-0.81, -0.09, 0.02],
    #      [0.74, -0.91, -0.60], [-0.82, -0.92, 0.71],
    #      [-0.26, 0.68, 0.15], [0.80, -0.94, -0.83]]
    # w = [0, 0, 0, -1, 0, 0, 2]
    # solver.Extreme_Learning_Machine(V, w, x)

    # tutorial 07 -- best sparse code
    # Vt = [[0.4, 0.55, 0.5, -0.1, -0.5, 0.9, 0.5, 0.45],
    #       [-0.6, -0.45, -0.5, 0.9, -0.5, 0.1, 0.5, 0.55]]
    # x = [[-0.05], [-0.95]]
    # yt = [1, 0, 0, 0, 1, 0, 0, 0]
    # yt = [0, 0, 1, 0, 0, 0, -1, 0]
    # yt = [0, 0, 0, -1, 0, 0, 0, 0]
    # solver.best_sparse_code(Vt, x, yt)

    # tutorial 08 -- SVM
    # x = np.transpose([[1, 1], [1, -1], [-1, 1], [-1, -1]])
    # y = [1, 1, -1, -1]
    # x = np.transpose([[5, 1], [5, -1], [3, 0]])
    # y = [1, 1, -1]
    # solver.SVM(x, y)
    # x = np.transpose([[1, 2], [7, 8]])
    # y = [1, -1]
    # solver.SVM(x, y)
    # x = np.transpose([[5, 1], [5, -1], [3, 0]])
    # y = [1, 1, -1]
    # solver.SVM(x, y)

    # tutorial 09 -- ENSEMBLE -- AdaBoost algorithm
    # x = np.transpose([[1, 0], [-1, 0], [0, 1], [0, -1]])
    # h = [[1, -1, -1, -1], [-1, 1, 1, 1], [1, -1, 1, 1], [-1, 1, -1, -1],
    #      [1, 1, -1, 1], [-1, -1, 1, -1], [-1, -1, -1, 1], [1, 1, 1, -1]]
    # epoch = 3
    # solver.adaBoost_algorithm(x, epoch, h)

    # tutorial 10 -- K-means algorithm
    # S = np.transpose([[-1, 3], [1, 4], [0, 5], [4, -1], [3, 0], [5, 1]])
    # m1 = np.transpose([-1, 3])
    # m2 = np.transpose([5, 1])
    # solver.Kmeans_algorithm(S,m1,m2)

    # tutorial 10 -- competitive learning algorithm (without normalisation)
    # S = np.transpose([[-1, 3], [1, 4], [0, 5], [4, -1], [3, 0], [5, 1]])
    # m1 = S[:, 0] / 2
    # m2 = S[:, 2] / 2
    # m3 = S[:, 4] / 2
    # eta = 0.1
    # epoch = 1
    # x = np.transpose([[0, 5], [-1, 3], [-1, 3], [3, 0], [5, 1]])
    # test_x = np.transpose([0, -2])
    # solver.competitive_learning_algorithm(S, m1, m2, m3, eta, epoch, x, test_x)

    # tutorial 10 -- basic leader follower algorithm (without normalisation)
    # S = np.transpose([[-1, 3], [1, 4], [0, 5], [4, -1], [3, 0], [5, 1]])
    # theta = 3
    # eta = 0.5
    # epoch = 1
    # x = np.transpose([[0, 5], [-1, 3], [-1, 3], [3, 0], [5, 1]])
    # test_x = np.transpose([0, -2])
    # solver.basic_leader_follower_algorithm(S, theta, eta, epoch, x, test_x)

    # tutorial 10 -- fuzzy K-means algorithm
    # S = np.transpose([[-1, 3], [1, 4], [0, 5], [4, -1], [3, 0], [5, 1]])
    # K = 2
    # b = 2
    # change = 0.5
    # mu = np.transpose([[1, 0], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0, 1]])
    # solver.fuzzy_Kmeans_algorithm(S, K, b, change, mu)

    # tutorial 10 -- agglomeration hierarchical algorithm
    # 这部分代码不👌，只是用来走通这个算法流程
    # 只适用以下数据 -- tutorial 10 最后一题
    # 主要思想是找到最小的欧几里德距离然后合并cluster
    # 合并后的列数据为最小的欧几里德距离
    # 合并进行c次
    # x = np.transpose([[-1, 3], [1, 2], [0, 1], [4, 0], [5, 4], [3, 2]])
    # # c = 3
    # solver.agglomeration_hierarchical_algorithm(x)
