import tensorflow as tf
from flip_gradient import flip_gradient
import time
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.gridspec as gridspec
import scipy.linalg as linalg
from threading import Thread

TRes_id_distributor = 0
plt.ion()

def change_lambda(ob):
    ob.lbda = float(input(""))
    thd = Thread(target=change_lambda, args=(ob, ))
    thd.daemon = True
    thd.start()

def test_domain_discrepancy(X, D, ax, method='LDA2'):
    np.random.seed(309597)
    ind = np.arange(len(X))
    np.random.shuffle(ind)
    y = np.argmax(D, axis=1)
    # clf = SVC(kernel='linear')
    # train_X = X[ind[:int(len(ind) * 0.7)]]
    # train_y = y[ind[:int(len(ind) * 0.7)]]
    # test_X = X[ind[int(len(ind) * 0.7):]]
    # test_y = y[ind[int(len(ind) * 0.7):]]
    # clf.fit(train_X, train_y)
    if method == 'PCA':
        pca = PCA(n_components=2)
        X_2dim = pca.fit_transform(X)
        ax.cla()
        ax.scatter(X_2dim[:, 0], X_2dim[:, 1], c=y)
    elif method == 'LDA':
        lda = LinearDiscriminantAnalysis()
        X_2dim = lda.fit(X, y).transform(X)
        ax.cla()
        ax.scatter(X_2dim[:, 0], np.arange(len(X_2dim)), c=y)
    elif method == 'LDA2':
        lda = LinearDiscriminantAnalysis()
        pca = PCA(n_components=1)
        # X_2dim = pca.fit_transform(X)
        X_2dim = np.concatenate((lda.fit(X, y).transform(X), pca.fit_transform(X)), axis=1)
        ax.cla()
        ax.scatter(X_2dim[:, 0], X_2dim[:, 1], c=y)
    else:
        ax.cla()
        ax.scatter(X[:, 0], X[:, 1], c=y)
    return -1.


class BatchGenerator:
    def __init__(self, *data_list, batch_size=400, seed=None):
        if seed is None:
            seed = int(time.time())
        np.random.seed(seed)
        self.data_list = data_list
        print([d.shape for d in self.data_list])
        self.epoch = 0
        self.tmp_pos = 0
        self.data_size = data_list[0].shape[0]
        self.batch_size = batch_size
        self.epoch_changed = True
        if self.data_size < self.batch_size:
            print("the batch size is too large!!")
            exit(-1)
        self.ind = np.arange(self.data_size)
        np.random.shuffle(self.ind)

    def get_batch(self):
        ret = [d[self.ind[self.tmp_pos:self.tmp_pos + self.batch_size]] for d in self.data_list]
        self.tmp_pos += self.batch_size
        if self.tmp_pos + self.batch_size > self.data_size:
            self.tmp_pos = 0
            np.random.shuffle(self.ind)
            self.epoch += 1
            self.epoch_changed = True
        else:
            self.epoch_changed = False
        return ret


class BalancedBatchGenerator:
    def __init__(self, balance_accordance, *data_list, batch_size=400, seed=None):
        data_list = [np.array(d) for d in data_list]
        self.indicator = np.argmax(data_list[balance_accordance], axis=1)
        self.buf = []
        self.class_num = data_list[balance_accordance].shape[1]
        for i in range(self.class_num):
            ind = [j for j, k in enumerate(self.indicator) if k == i]
            self.buf.append([np.array(d[ind]) for d in data_list])
        self.batches = [BatchGenerator(*b, batch_size=int(batch_size/self.class_num)) for b in self.buf]
        self.epoch = 0
        self.epoch_changed = True

    def get_batch(self):
        ret = [batch.get_batch() for batch in self.batches]
        ret = [np.concatenate([r[i] for r in ret], axis=0) for i in range(len(ret[0]))]
        self.epoch = np.min([batch.epoch for batch in self.batches])
        self.epoch_changed = self.batches[int(np.argmin([batch.epoch for batch in self.batches]))].epoch_changed
        return ret


def active(s="sigmoid"):
    if s == "sigmoid":
        return tf.nn.sigmoid
    elif s == "relu":
        return tf.nn.relu
    elif s == "elu":
        return tf.nn.elu
    elif s == "leaky_relu":
        return tf.nn.leaky_relu
    else:
        return tf.identity


def softmax(x):
    tmp = tf.exp(x)
    return tmp / tf.expand_dims(tf.reduce_sum(tmp, axis=1) + 0.01, axis=1)


def cross_entropy(label, logit):
    return - tf.reduce_sum(label * tf.log(logit + 0.0001)) / (0.0001 + tf.reduce_sum(label))


class FeatureExtractorBiasOnly:
    counter = 0

    def __init__(self, feature_size, domain_size, node_nums=(20, 20, 20), activation="sigmoid", name=None):
        self.node_nums = node_nums
        self.activation = activation
        self.id = FeatureExtractorBiasOnly.counter
        FeatureExtractorBiasOnly.counter += 1
        self.active = active(activation)
        if name is None:
            self.name = "/".join(["TresNet", "FE", "BiasOnly", str(self.id)])
        else:
            self.name = name
        self.Ws = []
        self.bs = []
        for i, n in enumerate(node_nums):
            if i == 0:
                self.Ws.append(tf.get_variable(name=self.name + "/W" + str(i),
                                               shape=[feature_size + domain_size, node_nums[i]],
                                               initializer=tf.contrib.layers.xavier_initializer()))
            else:
                self.Ws.append(tf.get_variable(name=self.name + "/W" + str(i),
                                               shape=[node_nums[i - 1] + domain_size, node_nums[i]],
                                               initializer=tf.contrib.layers.xavier_initializer()))
            # self.bs.append(tf.get_variable(name=self.name+"/b"+str(i), shape=[node_nums[i]],
            #                                initializer=tf.contrib.layers.xavier_initializer()))
            self.bs.append(tf.Variable(initial_value=np.zeros(shape=[node_nums[i]]), name=self.name + "/b" + str(i),
                                       dtype=tf.float32))

    def __call__(self, X, D):
        """Generate a feature extractor with the parameters (tensors).
           Be aware the shape of the input (SampleNum by Dimensionality)"""
        tmpX = X
        for W, b in zip(self.Ws, self.bs):
            tmpX = tf.concat((tmpX, D), axis=1)
            tmpX = self.active(tf.matmul(tmpX, W) + b)
        return tmpX


class FeatureExtractorFC:
    counter = 0

    def __init__(self, feature_size, domain_size, node_nums=(20, 20, 20), activation="sigmoid", name=None, lN=2):
        self.node_nums = node_nums
        self.activation = activation
        self.id = FeatureExtractorBiasOnly.counter
        FeatureExtractorBiasOnly.counter += 1
        self.active = active(activation)
        if name is None:
            self.name = "/".join(["TresNet", "FE", "FC", str(self.id)])
        else:
            self.name = name
        self.Ws = []
        self.bs = []
        self.resWs = []
        self.resbs = []
        self.regularizer = 0
        para_num = 0
        for i, n in enumerate(node_nums):
            if i == 0:
                self.resWs.append(tf.get_variable(name=self.name + "/W_res_" + str(i),
                                                  shape=[domain_size, feature_size, node_nums[i]],
                                                  initializer=tf.contrib.layers.xavier_initializer()))
                para_num += feature_size * node_nums[i]
            else:
                self.resWs.append(tf.get_variable(name=self.name + "/W_res_" + str(i),
                                                  shape=[domain_size, node_nums[i - 1], node_nums[i]],
                                                  initializer=tf.contrib.layers.xavier_initializer()))
                para_num += node_nums[i - 1] * node_nums[i]
            self.resbs.append(tf.get_variable(name=self.name + "/b_res_" + str(i),
                                              shape=[domain_size, node_nums[i]],
                                              initializer=tf.contrib.layers.xavier_initializer()))
            if lN == 1:
                self.regularizer += tf.reduce_sum(tf.abs(self.resWs[-1]))
                self.regularizer += tf.reduce_sum(tf.abs(self.resbs[-1]))
            else:
                self.regularizer += tf.reduce_sum(tf.square(self.resWs[-1]))
                self.regularizer += tf.reduce_sum(tf.square(self.resbs[-1]))
        self.regularizer /= float((np.sum(node_nums) + para_num) * domain_size)
        for i, n in enumerate(node_nums):
            if i == 0:
                self.Ws.append(tf.get_variable(name=self.name + "/W" + str(i),
                                               shape=[feature_size, node_nums[i]],
                                               initializer=tf.contrib.layers.xavier_initializer()))
            else:
                self.Ws.append(tf.get_variable(name=self.name + "/W" + str(i),
                                               shape=[node_nums[i - 1], node_nums[i]],
                                               initializer=tf.contrib.layers.xavier_initializer()))
            self.bs.append(tf.get_variable(name=self.name + "/b" + str(i), shape=[node_nums[i]],
                                           initializer=tf.contrib.layers.xavier_initializer()))

    def __call__(self, X, D_inds):
        """Generate a feature extractor with the parameters (tensors).
           Be aware the shape of the input (SampleNum by Dimensionality)"""
        tmpX = X
        for W, b, resW, resb in zip(self.Ws, self.bs, self.resWs, self.resbs):
            # tmpX = tf.concat((tmpX, D), axis=1)
            tmpX = self.active(tf.matmul(tmpX, W) + b + tf.einsum('ij,ijk->ik',
                                                                  tmpX,
                                                                  tf.gather(resW, D_inds, axis=0)) + tf.gather(resb,
                                                                                                               D_inds,
                                                                                                               axis=0))
        return tmpX, self.regularizer


class FeatureExtractorPC:
    """residual nodes are partially connected to introduce solid regularization"""
    counter = 0

    def __init__(self, feature_size, domain_size, node_nums=(20, 20, 20), connect_portion=0.3, activation="sigmoid",
                 name=None, lN=2):
        self.node_nums = node_nums
        self.activation = activation
        self.id = FeatureExtractorBiasOnly.counter
        FeatureExtractorBiasOnly.counter += 1
        self.active = active(activation)
        if name is None:
            self.name = "/".join(["TresNet", "FE", "PC", str(self.id)])
        else:
            self.name = name
        self.Ws = []
        self.bs = []
        self.resWs = []
        self.resbs = []
        self.assist_mat = tf.constant([[1, 0], [0, 1], [0, 0], [0, -1]])
        self.regularizer = 0
        para_num = 0
        for i, n in enumerate(node_nums):
            if i == 0:
                self.resWs.append(tf.get_variable(name=self.name + "/W_res_" + str(i),
                                                  shape=[domain_size, feature_size,
                                                         int(node_nums[i] * connect_portion)],
                                                  initializer=tf.contrib.layers.xavier_initializer()))
                para_num += feature_size * int(node_nums[i] * connect_portion)
            else:
                self.resWs.append(tf.get_variable(name=self.name + "/W_res_" + str(i),
                                                  shape=[domain_size, node_nums[i - 1],
                                                         int(node_nums[i] * connect_portion)],
                                                  initializer=tf.contrib.layers.xavier_initializer()))
                para_num += node_nums[i - 1] * int(node_nums[i] * connect_portion)
            self.resbs.append(tf.get_variable(name=self.name + "/b_res_" + str(i),
                                              shape=[domain_size, int(node_nums[i] * connect_portion)],
                                              initializer=tf.contrib.layers.xavier_initializer()))
            if lN == 1:
                self.regularizer += tf.reduce_sum(tf.abs(self.resWs[-1]))
                self.regularizer += tf.reduce_sum(tf.abs(self.resbs[-1]))
            else:
                self.regularizer += tf.reduce_sum(tf.square(self.resWs[-1]))
                self.regularizer += tf.reduce_sum(tf.square(self.resbs[-1]))
        self.regularizer /= float((np.sum([int(nn * connect_portion) for nn in node_nums]) + para_num) * domain_size)
        for i, n in enumerate(node_nums):
            if i == 0:
                self.Ws.append(tf.get_variable(name=self.name + "/W" + str(i),
                                               shape=[feature_size, node_nums[i]],
                                               initializer=tf.contrib.layers.xavier_initializer()))
            else:
                self.Ws.append(tf.get_variable(name=self.name + "/W" + str(i),
                                               shape=[node_nums[i - 1], node_nums[i]],
                                               initializer=tf.contrib.layers.xavier_initializer()))
            self.bs.append(tf.get_variable(name=self.name + "/b" + str(i), shape=[node_nums[i]],
                                           initializer=tf.contrib.layers.xavier_initializer()))

    def __call__(self, X, D_inds):
        """Generate a feature extractor with the parameters (tensors).
           Be aware the shape of the input (SampleNum by Dimensionality)"""
        tmpX = X
        for W, b, resW, resb in zip(self.Ws, self.bs, self.resWs, self.resbs):
            # tmpX = tf.concat((tmpX, D), axis=1)
            new_tmpX = self.active(tf.matmul(tmpX, W) + b)
            tmp_res = tf.einsum('ij,ijk->ik', tmpX, tf.gather(resW, D_inds, axis=0)) + tf.gather(resb, D_inds, axis=0)
            col_compensation = tf.zeros(
                tf.reshape(
                    tf.matmul(
                        tf.expand_dims(
                            tf.concat((tf.shape(new_tmpX), tf.shape(tmp_res)), axis=0),
                            axis=0),
                        self.assist_mat),
                    shape=(-1,)),
                dtype=tf.float32)
            tmpX = new_tmpX + tf.concat((tmp_res, col_compensation), axis=1)
        return tmpX, self.regularizer


class FeatureExtractorBiasOnlyRegularized:
    counter = 0

    def __init__(self, feature_size, domain_size, node_nums=(20, 20, 20), activation="sigmoid", name=None, lN=2):
        self.node_nums = node_nums
        self.activation = activation
        self.id = FeatureExtractorBiasOnly.counter
        FeatureExtractorBiasOnly.counter += 1
        self.active = active(activation)
        if name is None:
            self.name = "/".join(["TresNet", "FE", "BiasOnly", str(self.id)])
        else:
            self.name = name
        self.Ws = []
        self.bs = []
        self.regularizer = 0

        for i, n in enumerate(node_nums):
            if i == 0:
                self.Ws.append(tf.get_variable(name=self.name + "/W" + str(i),
                                               shape=[feature_size + domain_size, node_nums[i]],
                                               initializer=tf.contrib.layers.xavier_initializer()))
                if lN == 1:
                    self.regularizer = self.regularizer + tf.reduce_sum(tf.abs(tf.slice(self.Ws[-1],
                                                                                        (feature_size, 0),
                                                                                        (-1, -1))))
                else:
                    self.regularizer = self.regularizer + tf.reduce_sum(tf.square(tf.slice(self.Ws[-1],
                                                                                           (feature_size, 0),
                                                                                           (-1, -1))))
            else:
                self.Ws.append(tf.get_variable(name=self.name + "/W" + str(i),
                                               shape=[node_nums[i - 1] + domain_size, node_nums[i]],
                                               initializer=tf.contrib.layers.xavier_initializer()))
                if lN == 1:
                    self.regularizer = self.regularizer + tf.reduce_sum(tf.square(tf.slice(self.Ws[-1],
                                                                                           (node_nums[i], 0),
                                                                                           (-1, -1))))
                else:
                    self.regularizer = self.regularizer + tf.reduce_sum(tf.square(tf.slice(self.Ws[-1],
                                                                                           (node_nums[i], 0),
                                                                                           (-1, -1))))
            # self.bs.append(tf.get_variable(name=self.name+"/b"+str(i), shape=[node_nums[i]],
            #                                initializer=tf.contrib.layers.xavier_initializer()))
            self.bs.append(tf.Variable(initial_value=np.zeros(shape=[node_nums[i]]), name=self.name + "/b" + str(i),
                                       dtype=tf.float32))
        self.regularizer = self.regularizer / float(np.sum(node_nums) * domain_size)

    def __call__(self, X, D):
        """Generate a feature extractor with the parameters (tensors).
           Be aware the shape of the input (SampleNum by Dimensionality)"""
        tmpX = X
        for W, b in zip(self.Ws, self.bs):
            tmpX = tf.concat((tmpX, D), axis=1)
            tmpX = self.active(tf.matmul(tmpX, W) + b)
        return tmpX, self.regularizer


class FeatureExtractorBiasOnlyRegularized_v2:
    counter = 0

    def __init__(self, feature_size, domain_size, node_nums=(20, 20, 20), activation="sigmoid", name=None, lN=2):
        self.node_nums = node_nums
        self.activation = activation
        self.id = FeatureExtractorBiasOnly.counter
        FeatureExtractorBiasOnly.counter += 1
        self.active = active(activation)
        if name is None:
            self.name = "/".join(["TresNet", "FE", "BiasOnly", str(self.id)])
        else:
            self.name = name
        self.Ws = []
        self.bs = []
        self.resbs = []
        self.regularizer = 0
        for i, n in enumerate(node_nums):
            self.resbs.append(tf.Variable(np.zeros([domain_size, node_nums[i]]), name=self.name + "/b_res_" + str(i),
                                          dtype=tf.float32))
            if lN == 1:
                self.regularizer += tf.reduce_sum(tf.abs(self.resbs[-1]))
            else:
                self.regularizer += tf.reduce_sum(tf.square(self.resbs[-1]))
        self.regularizer = self.regularizer / (np.sum(node_nums) * domain_size)
        for i, n in enumerate(node_nums):
            if i == 0:
                self.Ws.append(tf.get_variable(name=self.name + "/W" + str(i),
                                               shape=[feature_size, node_nums[i]],
                                               initializer=tf.contrib.layers.xavier_initializer()))
            else:
                self.Ws.append(tf.get_variable(name=self.name + "/W" + str(i),
                                               shape=[node_nums[i - 1], node_nums[i]],
                                               initializer=tf.contrib.layers.xavier_initializer()))
            self.bs.append(tf.Variable(initial_value=np.zeros(shape=[node_nums[i]]), name=self.name + "/b" + str(i),
                                       dtype=tf.float32))

    def __call__(self, X, D):
        """Generate a feature extractor with the parameters (tensors).
           Be aware the shape of the input (SampleNum by Dimensionality)"""
        tmpX = X
        D_ind = tf.reshape(tf.slice(tf.where(tf.not_equal(D, tf.constant(0.0, dtype=tf.float32))),
                                    begin=(0, 1), size=(-1, -1)), shape=(-1,))
        for W, b, resb in zip(self.Ws, self.bs, self.resbs):
            tmpX = self.active(tf.matmul(tmpX, W) + b + tf.gather(resb, indices=D_ind))
        return tmpX, self.regularizer


class FeatureExtractor:
    counter = 0

    def __init__(self, feature_size, domain_size,
                 node_nums=(20, 20, 20),
                 activation="sigmoid",
                 name=None,
                 lN=2,
                 residual_type='b'):
        self.node_nums = node_nums
        self.activation = activation
        self.res_type = residual_type
        self.id = FeatureExtractorBiasOnly.counter
        FeatureExtractorBiasOnly.counter += 1
        self.active = active(activation)
        if name is None:
            self.name = "/".join(["TresNet", "FE", residual_type, str(self.id)])
        else:
            self.name = name
        self.Ws = []
        self.bs = []
        self.resWs = []
        self.resbs = []
        self.regularizer = tf.constant(0.0, dtype=tf.float32)
        para_num = 0
        for i, n in enumerate(node_nums):
            if 'w' in residual_type.lower():
                if i == 0:
                    self.resWs.append(tf.Variable(initial_value=np.zeros([domain_size, feature_size, node_nums[i]]),
                                                  name=self.name + "/W_res_" + str(i), dtype=tf.float32))
                    # self.resWs.append(tf.get_variable(name=self.name + "/W_res_" + str(i),
                    #                                   shape=[domain_size, feature_size, node_nums[i]],
                    #                                   initializer=tf.contrib.layers.xavier_initializer()))
                    para_num += feature_size * node_nums[i]
                else:
                    self.resWs.append(tf.Variable(initial_value=np.zeros([domain_size, node_nums[i - 1], node_nums[i]]),
                                                  name=self.name + "/W_res_" + str(i), dtype=tf.float32))
                    # self.resWs.append(tf.get_variable(name=self.name + "/W_res_" + str(i),
                    #                                   shape=[domain_size, node_nums[i - 1], node_nums[i]],
                    #                                   initializer=tf.contrib.layers.xavier_initializer()))
                    para_num += node_nums[i - 1] * node_nums[i]
                self.regularizer += tf.reduce_sum(tf.pow(tf.abs(self.resWs[-1]), lN))
            if 'b' in residual_type.lower():
                # self.resbs.append(tf.get_variable(name=self.name + "/b_res_" + str(i),
                #                                   shape=[domain_size, node_nums[i]],
                #                                   initializer=tf.contrib.layers.xavier_initializer()))
                self.resbs.append(tf.Variable(name=self.name + "/b_res_" + str(i),
                                              initial_value=np.zeros([domain_size, node_nums[i]]), dtype=tf.float32))
                # self.regularizer += tf.reduce_sum(tf.pow(tf.abs(self.resbs[-1]), lN))
        self.regularizer /= float(para_num * domain_size)
        for i, n in enumerate(node_nums):
            if i == 0:
                self.Ws.append(tf.get_variable(name=self.name + "/W" + str(i),
                                               shape=[feature_size, node_nums[i]],
                                               initializer=tf.contrib.layers.xavier_initializer()))
            else:
                self.Ws.append(tf.get_variable(name=self.name + "/W" + str(i),
                                               shape=[node_nums[i - 1], node_nums[i]],
                                               initializer=tf.contrib.layers.xavier_initializer()))
            self.bs.append(tf.get_variable(name=self.name + "/b" + str(i), shape=[node_nums[i]],
                                           initializer=tf.contrib.layers.xavier_initializer()))

    def __call__(self, X, D):
        """Generate a feature extractor with the parameters (tensors).
           Be aware the shape of the input (SampleNum by Dimensionality)"""
        tmpX = X
        D_ind = tf.reshape(tf.slice(tf.where(tf.not_equal(D, tf.constant(0.0, dtype=tf.float32))),
                                    begin=(0, 1), size=(-1, -1)), shape=(-1,))
        for i, W, b in zip(np.arange(len(self.Ws)), self.Ws, self.bs):
            # tmpX = tf.concat((tmpX, D), axis=1)
            ttmpX = tf.matmul(tmpX, W) + b
            if 'w' in self.res_type.lower():
                ttmpX += tf.einsum('ij,ijk->ik', tmpX, tf.gather(self.resWs[i], D_ind, axis=0))
            if 'b' in self.res_type.lower():
                ttmpX += tf.gather(self.resbs[i], D_ind, axis=0)
            tmpX = self.active(ttmpX)
        return tmpX, self.regularizer


class Discriminator:
    counter = 0

    def __init__(self, feature_size, node_nums=(20, 10, 2), activation="sigmoid", name=None):
        """Note that the last integer in node_nums indicates how many classes there should be"""
        self.node_nums = node_nums
        self.activation = activation
        self.id = Discriminator.counter
        Discriminator.counter += 1
        self.active = active(activation)
        if name is None:
            self.name = "/".join(["TresNet", "Discriminator", str(self.id)])
        else:
            self.name = name
        self.Ws = []
        self.bs = []
        for i, n in enumerate(node_nums):
            if i == 0:
                self.Ws.append(tf.get_variable(name=self.name + "/W" + str(i),
                                               shape=[feature_size, node_nums[i]],
                                               initializer=tf.contrib.layers.xavier_initializer()))
            else:
                self.Ws.append(tf.get_variable(name=self.name + "/W" + str(i),
                                               shape=[node_nums[i - 1], node_nums[i]],
                                               initializer=tf.contrib.layers.xavier_initializer()))
            # self.bs.append(tf.get_variable(name=self.name+"/b"+str(i), shape=[node_nums[i]],
            #                                initializer=tf.contrib.layers.xavier_initializer()))
            self.bs.append(tf.Variable(initial_value=np.zeros(shape=[node_nums[i]]), name=self.name + "/b" + str(i),
                                       dtype=tf.float32))

    def __call__(self, X):
        tmpX = X
        for W, b in zip(self.Ws[:-1], self.bs[:-1]):
            tmpX = self.active(tf.matmul(tmpX, W) + b)
        if len(self.Ws) > 0:
            tmpX = tf.matmul(tmpX, self.Ws[-1]) + self.bs[-1]
        return tmpX


class Regressor:
    counter = 0

    def __init__(self, feature_size, node_nums=(20, 10, 1), activation="sigmoid", name=None):
        """Note that the last integer in node_nums indicates the dimensionality of the labels"""
        self.node_nums = node_nums
        self.activation = activation
        self.id = Regressor.counter
        Regressor.counter += 1
        self.active = active(activation)
        if name is None:
            self.name = "/".join(["TresNet", "Regressor", str(self.id)])
        else:
            self.name = name
        self.Ws = []
        self.bs = []
        for i, n in enumerate(node_nums):
            if i == 0:
                self.Ws.append(tf.get_variable(name=self.name + "/W" + str(i),
                                               shape=[feature_size, node_nums[i]],
                                               initializer=tf.contrib.layers.xavier_initializer()))
            else:
                self.Ws.append(tf.get_variable(name=self.name + "/W" + str(i),
                                               shape=[node_nums[i - 1], node_nums[i]],
                                               initializer=tf.contrib.layers.xavier_initializer()))
            self.bs.append(tf.get_variable(name=self.name + "/b" + str(i), shape=[node_nums[i]],
                                           initializer=tf.contrib.layers.xavier_initializer()))

    def __call__(self, X):
        """the last layer always use identity activation functions"""
        tmpX = X
        for W, b in zip(self.Ws[:-1], self.bs[:-1]):
            tmpX = self.active(tf.matmul(tmpX, W) + b)
        if len(self.Ws) > 0:
            tmpX = tf.matmul(tmpX, self.Ws[-1]) + self.bs[-1]
        return tmpX


class TResNetRegressProto:
    def __init__(self, feature_size, domain_size, label_size=1, lbda=0.5, lr=1e-5, alpha=0, max_flip_ratio=0.01, lN=2, id=-1):
        self.feature_size = feature_size
        self.domain_size = domain_size
        self.label_size = label_size
        self.max_flip_ratio = max_flip_ratio
        self.last_dl = None
        self.id = id
        self.active = 'leaky_relu'
        self.lbda = lbda
        # self.fe = FeatureExtractorBiasOnlyRegularized_v2(feature_size=feature_size,
        #                                                  domain_size=domain_size,
        #                                                  node_nums=(80, 20,), lN=lN, activation=self.active)
        # self.dd = Discriminator(feature_size=20, node_nums=(10, domain_size,), activation=self.active)
        # self.cl = Discriminator(feature_size=20, node_nums=(10, label_size,), activation=self.active)
        self.fe = FeatureExtractorBiasOnlyRegularized_v2(feature_size=feature_size,
                                                         domain_size=domain_size,
                                                         node_nums=(80, 40, 20,), lN=lN, activation=self.active)
        self.dd = Discriminator(feature_size=20, node_nums=(domain_size,), activation=self.active)
        self.lp = Regressor(feature_size=20, node_nums=(label_size,), activation=self.active)
        # self.fe = FeatureExtractorBiasOnlyRegularized(feature_size=feature_size,
        #                                               domain_size=domain_size,
        #                                               node_nums=(120, 80, 40), lN=lN, activation=self.active)
        # self.dd = Discriminator(feature_size=40, node_nums=(20, 10, domain_size), activation=self.active)
        # self.cl = Discriminator(feature_size=40, node_nums=(20, 10, label_size), activation=self.active)
        self.tf_input_feaure = tf.placeholder(dtype=tf.float32, shape=(None, feature_size))
        self.tf_input_domain_feature = tf.placeholder(dtype=tf.float32, shape=(None, domain_size))
        self.tf_label = tf.placeholder(dtype=tf.float32, shape=(None, label_size))

        self.tf_lbda = tf.placeholder(dtype=tf.float32)
        self.tf_flip_gradient_ratio = tf.placeholder(dtype=tf.float32)

        self.tf_feature_embedding, self.regularizer = self.fe(self.tf_input_feaure, self.tf_input_domain_feature)
        self.tf_source_ind = tf.where(tf.not_equal(tf.reduce_sum(self.tf_label, axis=1), tf.constant(0.0, tf.float32)))
        self.tf_source_ind = tf.reshape(self.tf_source_ind, shape=[-1])
        self.tf_label_prediction = self.lp(tf.gather(self.tf_feature_embedding, self.tf_source_ind, axis=0))
        # self.tf_domain_prediction = self.dd(self.tf_feature_embedding)
        self.tf_domain_prediction = self.dd(flip_gradient(self.tf_feature_embedding,
                                                          l=self.tf_flip_gradient_ratio))

        # self.tf_prediction_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        #     labels=tf.gather(self.tf_label, self.tf_source_ind, axis=0),
        #     logits=self.tf_label_prediction))
        self.tf_prediction_loss = tf.reduce_mean(tf.square(self.tf_label_prediction - self.tf_label))
        self.tf_discrimination_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=self.tf_input_domain_feature, logits=self.tf_domain_prediction))

        self.balanced_loss = (1 - self.tf_lbda) * self.tf_prediction_loss + self.tf_lbda * self.tf_discrimination_loss
        self.regularized_loss = (1 - alpha) * self.balanced_loss + alpha * self.regularizer
        self.tf_optimize = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(self.regularized_loss)
        # tf.train.MomentumOptimizer()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.dataset = None
        self.dl_buf = []
        self.ad_train_switch = False

    def get_flip_ratio(self):
        return self.max_flip_ratio
        # if self.last_dl is None:
        #     return 0.
        # self.dl_buf.append(self.last_dl)
        # if len(self.dl_buf) > 10:
        #     self.dl_buf = self.dl_buf[1:]
        # if (len(self.dl_buf) >= 10 and np.std(self.dl_buf) <= 0.0) or self.last_dl < 0.5:
        #     self.ad_train_switch = True
        # if self.ad_train_switch:
        #     if self.last_dl > np.log(self.domain_size):
        #         self.ad_train_switch = False
        #         return 0
        #     return np.max([0., self.max_flip_ratio - self.last_dl * self.max_flip_ratio / np.log(self.domain_size)])
        # return 0

    def get_flip_ratio_inverse(self):
        if self.last_dl is None:
            return 0.
        # if self.last_dl >= np.log(self.domain_size):
        #     return 0.
        return np.min((self.max_flip_ratio, (1.0/self.last_dl) - (1.0/np.log(self.domain_size))))

    def get_flip_ratio_log(self):
        if self.last_dl is None:
            return 0.
        # if self.last_dl >= np.log(self.domain_size):
        #     return 0.
        return np.min((self.max_flip_ratio, np.log(self.last_dl * np.log(self.domain_size))))

    def fit(self, X, D, Y, vX, vD, vY, tX, tD, tY, epoch=5000):
        """label matrix Y should have elements larger than 0. An element with -1 value is non-labeled entry"""
        # thd = Thread(target=change_lambda, args=(self, ))
        # thd.daemon = True
        # thd.start()
        # self.dataset = BalancedBatchGenerator(1, X, D, Y, batch_size=100)
        plt.close("all")
        self.dataset = BatchGenerator(X, D, Y, batch_size=100)
        train_regression_losses = []
        train_discrimination_losses = []
        validation_regression_losses = []
        validation_predictions = []
        test_regression_losses = []
        test_predictions = []
        fig = plt.figure(figsize=(12, 10), tight_layout=True)
        gs = gridspec.GridSpec(4, 3, height_ratios=(1., 0.7, 0.7, 0.7))
        ax = [fig.add_subplot(gs[0, i]) for i in range(3)]
        ax2 = [fig.add_subplot(gs[i+1, :]) for i in range(3)]
        # fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        # fig2, ax2 = plt.subplots(2, 1, figsize=(15, 5))
        plot_discriminate_loss = []
        plot_train_label_prediction_loss = []
        plot_validate_label_prediction_loss = []
        plot_test_label_prediction_loss = []
        plot_test_label_prediction_acc = []
        while self.dataset.epoch < epoch:
            if self.dataset.epoch_changed:
                if self.dataset.epoch % 10 == 0:
                    print("\n*********************\n", "module", self.id, "epoch", self.dataset.epoch)
                rl, fe, dl, re, pr = self.sess.run((self.tf_prediction_loss, self.tf_feature_embedding,
                                                self.tf_discrimination_loss, self.regularizer, self.tf_label_prediction),
                                               feed_dict={self.tf_input_feaure: X,
                                                          self.tf_input_domain_feature: D,
                                                          self.tf_label: Y,
                                                          self.tf_flip_gradient_ratio: 0.1,
                                                          self.tf_lbda: self.lbda})
                train_regression_losses.append(rl)
                train_discrimination_losses.append(dl)
                if self.dataset.epoch % 10 == 0:
                    print("discriminate loss:", dl)
                    print("SVM discriminate acc:", test_domain_discrepancy(fe, D, ax[2]))
                    print("regularization norm:", np.sqrt(re))
                    print("train label prediction loss (RMSE):", np.sqrt(np.sqrt(rl)))
                    plot_discriminate_loss.append(dl)
                    plot_train_label_prediction_loss.append(np.sqrt(rl))
                if vX is not None:
                    rl, fe, pr = self.sess.run(
                        (self.tf_prediction_loss, self.tf_feature_embedding, self.tf_label_prediction),
                        feed_dict={self.tf_input_feaure: vX,
                                   self.tf_input_domain_feature: vD,
                                   self.tf_label: vY,
                                   self.tf_flip_gradient_ratio: 0.1,
                                   self.tf_lbda: self.lbda})
                    validation_regression_losses.append(rl)
                    validation_predictions = pr
                    if self.dataset.epoch % 10 == 0:
                        print("validate label prediction loss (RMSE):", np.sqrt(rl))
                        plot_validate_label_prediction_loss.append(np.sqrt(rl))
                if self.dataset.epoch % 10 == 0:
                    ax[0].cla()
                    ax[0].scatter(np.reshape(fe, newshape=(-1,)),
                                  np.arange(len(np.reshape(fe, newshape=(-1,)))))
                    ax[1].cla()
                    ax[1].scatter(np.reshape(pr, newshape=(-1,)),
                                  np.arange(len(np.reshape(pr, newshape=(-1,)))))
                rl, pr = self.sess.run((self.tf_prediction_loss, self.tf_label_prediction),
                                       feed_dict={self.tf_input_feaure: tX,
                                                  self.tf_input_domain_feature: tD,
                                                  self.tf_label: tY,
                                                  self.tf_flip_gradient_ratio: 0.1,
                                                  self.tf_lbda: self.lbda})
                test_regression_losses.append(rl)
                test_predictions = pr
                if self.dataset.epoch % 10 == 0:
                    print("test label prediction loss (RMSE):", np.sqrt(rl))
                    print("gradient flip ratio:", self.get_flip_ratio())
                    plot_test_label_prediction_loss.append(np.sqrt(rl))
                    ax2[0].cla()
                    ax2[1].cla()
                    ax2[2].cla()
                    ax2[2].axis(ymax=1.0, ymin=0.0)
                    ax2[0].plot(plot_discriminate_loss)
                    ax2[0].plot([0, len(plot_discriminate_loss)], [np.log(self.domain_size), np.log(self.domain_size)])
                    ax2[1].plot(plot_train_label_prediction_loss, c='r')
                    ax2[1].plot(plot_validate_label_prediction_loss, c='b')
                    ax2[1].plot(plot_test_label_prediction_loss, c='g')
                    ax2[2].plot(plot_test_label_prediction_acc)
                    plt.pause(0.01)

            x, d, y = self.dataset.get_batch()
            self.last_dl = self.sess.run(self.tf_discrimination_loss,
                              feed_dict={self.tf_input_feaure: X,
                                         self.tf_input_domain_feature: D,
                                         self.tf_label: Y,
                                         self.tf_flip_gradient_ratio: 0.1,
                                         self.tf_lbda: self.lbda})
            _ = self.sess.run(self.tf_optimize,
                              feed_dict={self.tf_input_feaure: x,
                                         self.tf_input_domain_feature: d,
                                         self.tf_label: y,
                                         self.tf_flip_gradient_ratio: self.get_flip_ratio(),
                                         self.tf_lbda: self.lbda})
        plt.close()
        # plt.figure("train regression MSE")
        # plt.plot(train_regression_losses)
        # plt.figure("train discrimination CE")
        # plt.plot(train_discrimination_losses)
        # plt.figure("validation regression MSE")
        # plt.plot(validation_regression_losses)
        # plt.figure("validation discrimination CE")
        # plt.plot(validation_discrimination_losses)
        return train_regression_losses, \
               train_discrimination_losses, \
               validation_regression_losses, \
               validation_predictions, \
               test_regression_losses, \
               test_predictions, \
               plot_discriminate_loss, \
               plot_train_label_prediction_loss, \
               plot_validate_label_prediction_loss,\
               plot_test_label_prediction_loss, \
               plot_test_label_prediction_acc, \
               ["train_regression_losses",
                "train_discrimination_losses",
                "validation_regression_losses",
                "validation_predictions",
                "test_regression_losses",
                "test_predictions",
                "plot_discriminate_loss",
                "plot_train_label_prediction_loss",
                "plot_validate_label_prediction_loss",
                "plot_test_label_prediction_loss",
                "plot_test_label_prediction_acc", ]


class TResNetClassifyProto:
    def __init__(self, feature_size, domain_size, label_size=2, lbda=0.5, lr=1e-5, alpha=0, max_flip_ratio=0.01, lN=2, id=-1):
        self.feature_size = feature_size
        self.domain_size = domain_size
        self.label_size = label_size
        self.max_flip_ratio = max_flip_ratio
        self.last_dl = None
        self.id = id
        self.active = 'leaky_relu'
        self.lbda = lbda
        # self.fe = FeatureExtractorBiasOnlyRegularized_v2(feature_size=feature_size,
        #                                                  domain_size=domain_size,
        #                                                  node_nums=(80, 20,), lN=lN, activation=self.active)
        # self.dd = Discriminator(feature_size=20, node_nums=(10, domain_size,), activation=self.active)
        # self.cl = Discriminator(feature_size=20, node_nums=(10, label_size,), activation=self.active)
        self.fe = FeatureExtractorBiasOnlyRegularized_v2(feature_size=feature_size,
                                                         domain_size=domain_size,
                                                         node_nums=(80, 40, 20,), lN=lN, activation=self.active)
        self.dd = Discriminator(feature_size=20, node_nums=(domain_size,), activation=self.active)
        self.lp = Discriminator(feature_size=20, node_nums=(label_size,), activation=self.active)
        # self.fe = FeatureExtractorBiasOnlyRegularized(feature_size=feature_size,
        #                                               domain_size=domain_size,
        #                                               node_nums=(120, 80, 40), lN=lN, activation=self.active)
        # self.dd = Discriminator(feature_size=40, node_nums=(20, 10, domain_size), activation=self.active)
        # self.cl = Discriminator(feature_size=40, node_nums=(20, 10, label_size), activation=self.active)
        self.tf_input_feaure = tf.placeholder(dtype=tf.float32, shape=(None, feature_size))
        self.tf_input_domain_feature = tf.placeholder(dtype=tf.float32, shape=(None, domain_size))
        self.tf_label = tf.placeholder(dtype=tf.float32, shape=(None, label_size))

        self.tf_lbda = tf.placeholder(dtype=tf.float32)
        self.tf_flip_gradient_ratio = tf.placeholder(dtype=tf.float32)

        self.tf_feature_embedding, self.regularizer = self.fe(self.tf_input_feaure, self.tf_input_domain_feature)
        self.tf_source_ind = tf.where(tf.not_equal(tf.reduce_sum(self.tf_label, axis=1), tf.constant(0.0, tf.float32)))
        self.tf_source_ind = tf.reshape(self.tf_source_ind, shape=[-1])
        self.tf_label_prediction = self.lp(tf.gather(self.tf_feature_embedding, self.tf_source_ind, axis=0))
        # self.tf_domain_prediction = self.dd(self.tf_feature_embedding)
        self.tf_domain_prediction = self.dd(flip_gradient(self.tf_feature_embedding,
                                                          l=self.tf_flip_gradient_ratio))

        self.tf_classification_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.gather(self.tf_label, self.tf_source_ind, axis=0),
            logits=self.tf_label_prediction))
        self.tf_discrimination_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=self.tf_input_domain_feature, logits=self.tf_domain_prediction))

        self.balanced_loss = (1 - self.tf_lbda) * self.tf_classification_loss + self.tf_lbda * self.tf_discrimination_loss
        self.regularized_loss = (1 - alpha) * self.balanced_loss + alpha * self.regularizer
        self.tf_optimize = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(self.regularized_loss)
        # tf.train.MomentumOptimizer()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.dataset = None
        self.dl_buf = []
        self.ad_train_switch = False

    def get_flip_ratio(self):
        return self.max_flip_ratio
        # if self.last_dl is None:
        #     return 0.
        # self.dl_buf.append(self.last_dl)
        # if len(self.dl_buf) > 10:
        #     self.dl_buf = self.dl_buf[1:]
        # if (len(self.dl_buf) >= 10 and np.std(self.dl_buf) <= 0.0) or self.last_dl < 0.5:
        #     self.ad_train_switch = True
        # if self.ad_train_switch:
        #     if self.last_dl > np.log(self.domain_size):
        #         self.ad_train_switch = False
        #         return 0
        #     return np.max([0., self.max_flip_ratio - self.last_dl * self.max_flip_ratio / np.log(self.domain_size)])
        # return 0

    def get_flip_ratio_inverse(self):
        if self.last_dl is None:
            return 0.
        # if self.last_dl >= np.log(self.domain_size):
        #     return 0.
        return np.min((self.max_flip_ratio, (1.0/self.last_dl) - (1.0/np.log(self.domain_size))))

    def get_flip_ratio_log(self):
        if self.last_dl is None:
            return 0.
        # if self.last_dl >= np.log(self.domain_size):
        #     return 0.
        return np.min((self.max_flip_ratio, np.log(self.last_dl * np.log(self.domain_size))))

    def fit(self, X, D, Y, vX, vD, vY, tX, tD, tY, epoch=5000):
        """label matrix Y should have elements larger than 0. An element with -1 value is non-labeled entry"""
        # thd = Thread(target=change_lambda, args=(self, ))
        # thd.daemon = True
        # thd.start()
        # self.dataset = BalancedBatchGenerator(1, X, D, Y, batch_size=100)
        plt.close("all")
        self.dataset = BatchGenerator(X, D, Y, batch_size=100)
        train_regression_losses = []
        train_discrimination_losses = []
        validation_regression_losses = []
        validation_predictions = []
        test_regression_losses = []
        test_predictions = []
        fig = plt.figure(figsize=(12, 10), tight_layout=True)
        gs = gridspec.GridSpec(4, 3, height_ratios=(1., 0.7, 0.7, 0.7))
        ax = [fig.add_subplot(gs[0, i]) for i in range(3)]
        ax2 = [fig.add_subplot(gs[i+1, :]) for i in range(3)]
        # fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        # fig2, ax2 = plt.subplots(2, 1, figsize=(15, 5))
        plot_discriminate_loss = []
        plot_train_label_prediction_loss = []
        plot_validate_label_prediction_loss = []
        plot_test_label_prediction_loss = []
        plot_test_label_prediction_acc = []
        while self.dataset.epoch < epoch:
            if self.dataset.epoch_changed:
                if self.dataset.epoch % 10 == 0:
                    print("\n*********************\n", "module", self.id, "epoch", self.dataset.epoch)
                rl, fe, dl, re, pr = self.sess.run((self.tf_classification_loss, self.tf_feature_embedding,
                                                self.tf_discrimination_loss, self.regularizer, self.tf_label_prediction),
                                               feed_dict={self.tf_input_feaure: X,
                                                          self.tf_input_domain_feature: D,
                                                          self.tf_label: Y,
                                                          self.tf_flip_gradient_ratio: 0.1,
                                                          self.tf_lbda: self.lbda})
                train_regression_losses.append(rl)
                train_discrimination_losses.append(dl)
                if self.dataset.epoch % 10 == 0:
                    print("discriminate loss:", dl)
                    print("SVM discriminate acc:", test_domain_discrepancy(fe, D, ax[2]))
                    print("regularization norm:", np.sqrt(re))
                    print("train label prediction loss:", rl)
                    plot_discriminate_loss.append(dl)
                    plot_train_label_prediction_loss.append(rl)
                if vX is not None:
                    rl, fe, pr = self.sess.run(
                        (self.tf_classification_loss, self.tf_feature_embedding, self.tf_label_prediction),
                        feed_dict={self.tf_input_feaure: vX,
                                   self.tf_input_domain_feature: vD,
                                   self.tf_label: vY,
                                   self.tf_flip_gradient_ratio: 0.1,
                                   self.tf_lbda: self.lbda})
                    validation_regression_losses.append(rl)
                    validation_predictions = pr
                    if self.dataset.epoch % 10 == 0:
                        acc = np.mean([1. if i == j else 0. for i, j in zip(np.argmax(pr, axis=1),
                                                                            np.argmax(vY, axis=1))])
                        print("validate label prediction loss:", rl)
                        print("validate label prediction acc:", acc)
                        plot_validate_label_prediction_loss.append(rl)
                if self.dataset.epoch % 10 == 0:
                    ax[0].cla()
                    ax[0].scatter(np.reshape(fe, newshape=(-1,)),
                                  np.arange(len(np.reshape(fe, newshape=(-1,)))))
                    ax[1].cla()
                    ax[1].scatter(np.reshape(pr, newshape=(-1,)),
                                  np.arange(len(np.reshape(pr, newshape=(-1,)))))
                rl, pr = self.sess.run((self.tf_classification_loss, self.tf_label_prediction),
                                       feed_dict={self.tf_input_feaure: tX,
                                                  self.tf_input_domain_feature: tD,
                                                  self.tf_label: tY,
                                                  self.tf_flip_gradient_ratio: 0.1,
                                                  self.tf_lbda: self.lbda})
                test_regression_losses.append(rl)
                test_predictions = pr
                if self.dataset.epoch % 10 == 0:
                    acc = np.mean([1. if i == j else 0. for i, j in zip(np.argmax(pr, axis=1),
                                                                        np.argmax(tY, axis=1))])
                    print("test label prediction loss:", rl)
                    print("test label prediction acc:", acc)
                    print("gradient flip ratio:", self.get_flip_ratio())
                    plot_test_label_prediction_loss.append(rl)
                    plot_test_label_prediction_acc.append(acc)
                    ax2[0].cla()
                    ax2[1].cla()
                    ax2[2].cla()
                    ax2[2].axis(ymax=1.0, ymin=0.0)
                    ax2[0].plot(plot_discriminate_loss)
                    ax2[0].plot([0, len(plot_discriminate_loss)], [np.log(self.domain_size), np.log(self.domain_size)])
                    ax2[1].plot(plot_train_label_prediction_loss, c='r')
                    ax2[1].plot(plot_validate_label_prediction_loss, c='b')
                    ax2[1].plot(plot_test_label_prediction_loss, c='g')
                    ax2[2].plot(plot_test_label_prediction_acc)
                    plt.pause(0.01)

            x, d, y = self.dataset.get_batch()
            self.last_dl = self.sess.run(self.tf_discrimination_loss,
                              feed_dict={self.tf_input_feaure: X,
                                         self.tf_input_domain_feature: D,
                                         self.tf_label: Y,
                                         self.tf_flip_gradient_ratio: 0.1,
                                         self.tf_lbda: self.lbda})
            _ = self.sess.run(self.tf_optimize,
                              feed_dict={self.tf_input_feaure: x,
                                         self.tf_input_domain_feature: d,
                                         self.tf_label: y,
                                         self.tf_flip_gradient_ratio: self.get_flip_ratio(),
                                         self.tf_lbda: self.lbda})
        plt.close()
        # plt.figure("train regression MSE")
        # plt.plot(train_regression_losses)
        # plt.figure("train discrimination CE")
        # plt.plot(train_discrimination_losses)
        # plt.figure("validation regression MSE")
        # plt.plot(validation_regression_losses)
        # plt.figure("validation discrimination CE")
        # plt.plot(validation_discrimination_losses)
        return train_regression_losses, \
               train_discrimination_losses, \
               validation_regression_losses, \
               validation_predictions, \
               test_regression_losses, \
               test_predictions, \
               plot_discriminate_loss, \
               plot_train_label_prediction_loss, \
               plot_validate_label_prediction_loss,\
               plot_test_label_prediction_loss, \
               plot_test_label_prediction_acc, \
               ["train_regression_losses",
                "train_discrimination_losses",
                "validation_regression_losses",
                "validation_predictions",
                "test_regression_losses",
                "test_predictions",
                "plot_discriminate_loss",
                "plot_train_label_prediction_loss",
                "plot_validate_label_prediction_loss",
                "plot_test_label_prediction_loss",
                "plot_test_label_prediction_acc", ]


# class TResNetRegressProto_v2:
#     def __init__(self, feature_size, domain_size, label_size=2, lbda=0.5, lr=1e-5, alpha=0, max_flip_ratio=0.01, lN=2, id=-1):
#         self.feature_size = feature_size
#         self.domain_size = domain_size
#         self.label_size = label_size
#         self.max_flip_ratio = max_flip_ratio
#         self.last_dl = None
#         self.id = id
#         self.active = 'leaky_relu'
#         self.lbda = lbda
#         # self.fe = FeatureExtractorBiasOnlyRegularized_v2(feature_size=feature_size,
#         #                                                  domain_size=domain_size,
#         #                                                  node_nums=(80, 20,), lN=lN, activation=self.active)
#         # self.dd = Discriminator(feature_size=20, node_nums=(10, domain_size,), activation=self.active)
#         # self.cl = Discriminator(feature_size=20, node_nums=(10, label_size,), activation=self.active)
#         self.fe = FeatureExtractorBiasOnlyRegularized_v2(feature_size=feature_size,
#                                                          domain_size=domain_size,
#                                                          node_nums=(100, 20,), lN=lN, activation=self.active)
#         self.dd = Discriminator(feature_size=20, node_nums=(domain_size,), activation=self.active)
#         self.lp = Regressor(feature_size=20, node_nums=(label_size,), activation=self.active)
#         # self.fe = FeatureExtractorBiasOnlyRegularized(feature_size=feature_size,
#         #                                               domain_size=domain_size,
#         #                                               node_nums=(120, 80, 40), lN=lN, activation=self.active)
#         # self.dd = Discriminator(feature_size=40, node_nums=(20, 10, domain_size), activation=self.active)
#         # self.cl = Discriminator(feature_size=40, node_nums=(20, 10, label_size), activation=self.active)
#         self.tf_input_feaure = tf.placeholder(dtype=tf.float32, shape=(None, feature_size))
#         self.tf_input_domain_feature = tf.placeholder(dtype=tf.float32, shape=(None, domain_size))
#         self.tf_label = tf.placeholder(dtype=tf.float32, shape=(None, label_size))
#
#         self.tf_lbda = tf.placeholder(dtype=tf.float32)
#         self.tf_flip_gradient_ratio = tf.placeholder(dtype=tf.float32)
#
#         self.tf_feature_embedding, self.regularizer = self.fe(self.tf_input_feaure, self.tf_input_domain_feature)
#         self.tf_source_ind = tf.where(tf.not_equal(tf.reduce_sum(self.tf_label, axis=1), tf.constant(0.0, tf.float32)))
#         self.tf_source_ind = tf.reshape(self.tf_source_ind, shape=[-1])
#         self.tf_label_prediction = self.lp(tf.gather(self.tf_feature_embedding, self.tf_source_ind, axis=0))
#         # self.tf_domain_prediction = self.dd(self.tf_feature_embedding)
#         self.tf_domain_prediction = self.dd(flip_gradient(self.tf_feature_embedding,
#                                                           l=self.tf_flip_gradient_ratio))
#
#         # self.tf_classification_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
#         #     labels=tf.gather(self.tf_label, self.tf_source_ind, axis=0),
#         #     logits=self.tf_label_prediction))
#         self.tf_prediction_loss = tf.reduce_mean(tf.square(tf.gather(self.tf_label, self.tf_source_ind, axis=0)
#                                                            - self.tf_label_prediction))
#         self.tf_discrimination_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
#             labels=self.tf_input_domain_feature, logits=self.tf_domain_prediction))
#
#         self.balanced_loss = (1 - self.tf_lbda) * self.tf_prediction_loss + self.tf_lbda * self.tf_discrimination_loss
#         self.regularized_loss = (1 - alpha) * self.balanced_loss + alpha * self.regularizer
#         self.tf_optimize = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(self.regularized_loss)
#         # tf.train.MomentumOptimizer()
#
#         config = tf.ConfigProto()
#         config.gpu_options.allow_growth = True
#         self.sess = tf.Session(config=config)
#         self.sess.run(tf.global_variables_initializer())
#         self.dataset = None
#         self.dl_buf = []
#         self.ad_train_switch = False
#
#     def get_flip_ratio(self):
#         if self.last_dl is None:
#             return 0.
#         self.dl_buf.append(self.last_dl)
#         if len(self.dl_buf) > 10:
#             self.dl_buf = self.dl_buf[1:]
#         if (len(self.dl_buf) >= 10 and np.std(self.dl_buf) <= 0.0) or self.last_dl < 0.5:
#             self.ad_train_switch = True
#         if self.ad_train_switch:
#             if self.last_dl > np.log(self.domain_size):
#                 self.ad_train_switch = False
#                 return 0
#             return np.max([0., self.max_flip_ratio - self.last_dl * self.max_flip_ratio / np.log(self.domain_size)])
#         return 0
#
#     def get_flip_ratio_inverse(self):
#         if self.last_dl is None:
#             return 0.
#         # if self.last_dl >= np.log(self.domain_size):
#         #     return 0.
#         return np.min((self.max_flip_ratio, (1.0/self.last_dl) - (1.0/np.log(self.domain_size))))
#
#     def get_flip_ratio_log(self):
#         if self.last_dl is None:
#             return 0.
#         # if self.last_dl >= np.log(self.domain_size):
#         #     return 0.
#         return np.min((self.max_flip_ratio, np.log(self.last_dl * np.log(self.domain_size))))
#
#     def fit(self, X, D, Y, vX, vD, vY, tX, tD, tY, epoch=8000):
#         """label matrix Y should have elements larger than 0. An element with -1 value is non-labeled entry"""
#         # thd = Thread(target=change_lambda, args=(self, ))
#         # thd.daemon = True
#         # thd.start()
#         self.dataset = BatchGenerator(X, D, Y, batch_size=20)
#         train_regression_losses = []
#         train_discrimination_losses = []
#         validation_regression_losses = []
#         validation_predictions = []
#         test_regression_losses = []
#         test_predictions = []
#         fig = plt.figure(figsize=(10, 6), tight_layout=True)
#         gs = gridspec.GridSpec(3, 3, height_ratios=(1., 0.7, 0.7))
#         ax = [fig.add_subplot(gs[0, i]) for i in range(3)]
#         ax2 = [fig.add_subplot(gs[i+1, :]) for i in range(2)]
#         # fig, ax = plt.subplots(1, 3, figsize=(15, 5))
#         # fig2, ax2 = plt.subplots(2, 1, figsize=(15, 5))
#         plot_discriminate_loss = []
#         plot_train_label_prediction_loss = []
#         plot_validate_label_prediction_loss = []
#         plot_test_label_prediction_loss = []
#         plot_test_label_prediction_acc = []
#         while self.dataset.epoch < epoch:
#             if self.dataset.epoch_changed:
#                 if self.dataset.epoch % 10 == 0:
#                     print("\n*********************\n", "module", self.id, "epoch", self.dataset.epoch)
#                 pl, fe, dl, re = self.sess.run((self.tf_prediction_loss, self.tf_feature_embedding,
#                                                 self.tf_discrimination_loss, self.regularizer),
#                                                feed_dict={self.tf_input_feaure: X,
#                                                           self.tf_input_domain_feature: D,
#                                                           self.tf_label: Y,
#                                                           self.tf_flip_gradient_ratio: 0.1,
#                                                           self.tf_lbda: self.lbda})
#                 pl = np.sqrt(pl)
#                 train_regression_losses.append(pl)
#                 train_discrimination_losses.append(dl)
#                 if self.dataset.epoch % 10 == 0:
#                     print("discriminate loss:", dl)
#                     print("SVM discriminate acc:", test_domain_discrepancy(fe, D, ax[2]))
#                     print("regularization norm:", np.sqrt(re))
#                     print("train label prediction loss:", pl)
#                     plot_discriminate_loss.append(dl)
#                     plot_train_label_prediction_loss.append(pl)
#                 pl, fe, pr = self.sess.run(
#                     (self.tf_prediction_loss, self.tf_feature_embedding, self.tf_label_prediction),
#                     feed_dict={self.tf_input_feaure: vX,
#                                self.tf_input_domain_feature: vD,
#                                self.tf_label: vY,
#                                self.tf_flip_gradient_ratio: 0.1,
#                                self.tf_lbda: self.lbda})
#                 pl = np.sqrt(pl)
#                 validation_regression_losses.append(pl)
#                 validation_predictions = pr
#                 if self.dataset.epoch % 10 == 0:
#                     acc = np.mean([1. if i == j else 0. for i, j in zip(np.argmax(pr, axis=1),
#                                                                         np.argmax(vY, axis=1))])
#                     print("validate label prediction loss:", pl)
#                     print("validate label prediction acc:", acc)
#                     plot_validate_label_prediction_loss.append(pl)
#                     ax[0].cla()
#                     ax[0].scatter(np.reshape(fe, newshape=(-1,)),
#                                   np.arange(len(np.reshape(fe, newshape=(-1,)))))
#                     ax[1].cla()
#                     ax[1].scatter(np.reshape(pr, newshape=(-1,)),
#                                   np.arange(len(np.reshape(pr, newshape=(-1,)))))
#                 pl, pr = self.sess.run((self.tf_prediction_loss, self.tf_label_prediction),
#                                        feed_dict={self.tf_input_feaure: tX,
#                                                   self.tf_input_domain_feature: tD,
#                                                   self.tf_label: tY,
#                                                   self.tf_flip_gradient_ratio: 0.1,
#                                                   self.tf_lbda: self.lbda})
#                 pl = np.sqrt(pl)
#                 test_regression_losses.append(pl)
#                 test_predictions = pr
#                 if self.dataset.epoch % 10 == 0:
#                     print(pr, tY)
#                     print("test label prediction loss:", pl)
#                     print("gradient flip ratio:", self.get_flip_ratio())
#                     plot_test_label_prediction_loss.append(pl)
#                     ax2[0].cla()
#                     ax2[1].cla()
#                     ax2[0].plot(plot_discriminate_loss)
#                     ax2[0].plot([0, len(plot_discriminate_loss)], [np.log(self.domain_size), np.log(self.domain_size)])
#                     ax2[1].plot(plot_train_label_prediction_loss, c='r')
#                     ax2[1].plot(plot_validate_label_prediction_loss, c='b')
#                     ax2[1].plot(plot_test_label_prediction_loss, c='g')
#                     plt.pause(0.01)
#
#             x, d, y = self.dataset.get_batch()
#             self.last_dl = self.sess.run(self.tf_discrimination_loss,
#                               feed_dict={self.tf_input_feaure: X,
#                                          self.tf_input_domain_feature: D,
#                                          self.tf_label: Y,
#                                          self.tf_flip_gradient_ratio: 0.1,
#                                          self.tf_lbda: self.lbda})
#             _ = self.sess.run(self.tf_optimize,
#                               feed_dict={self.tf_input_feaure: x,
#                                          self.tf_input_domain_feature: d,
#                                          self.tf_label: y,
#                                          self.tf_flip_gradient_ratio: self.get_flip_ratio(),
#                                          self.tf_lbda: self.lbda})
#         plt.close()
#         # plt.figure("train regression MSE")
#         # plt.plot(train_regression_losses)
#         # plt.figure("train discrimination CE")
#         # plt.plot(train_discrimination_losses)
#         # plt.figure("validation regression MSE")
#         # plt.plot(validation_regression_losses)
#         # plt.figure("validation discrimination CE")
#         # plt.plot(validation_discrimination_losses)
#         return train_regression_losses, \
#                train_discrimination_losses, \
#                validation_regression_losses, \
#                validation_predictions, \
#                test_regression_losses, \
#                test_predictions, \
#                plot_discriminate_loss, \
#                plot_train_label_prediction_loss, \
#                plot_validate_label_prediction_loss,\
#                plot_test_label_prediction_loss, \
#                plot_test_label_prediction_acc
#
#
# class TResNetRegressFC:
#     def __init__(self, feature_size, domain_size, label_size=1, lbda=0.5, lr=1e-5, alpha=0, lN=2):
#         self.feature_size = feature_size
#         self.domain_size = domain_size
#         self.label_size = label_size
#
#         self.fe = FeatureExtractorFC(feature_size=feature_size,
#                                      domain_size=domain_size,
#                                      node_nums=(80, 40), lN=lN)
#         self.dd = Discriminator(feature_size=40, node_nums=(10, domain_size))
#         self.rg = Regressor(feature_size=40, node_nums=(20, 10, label_size))
#         self.tf_input_feaure = tf.placeholder(dtype=tf.float32, shape=(None, feature_size))
#         self.tf_input_domain_feature = tf.placeholder(dtype=tf.float32, shape=(None, domain_size))
#         self.tf_domain_index = tf.argmax(self.tf_input_domain_feature, axis=1)
#         self.tf_label = tf.placeholder(dtype=tf.float32, shape=(None, label_size))
#         self.tf_feature_embedding, self.regularizer = self.fe(self.tf_input_feaure, self.tf_domain_index)
#         self.tf_label_prediction = self.rg(self.tf_feature_embedding)
#         self.tf_domain_prediction = self.dd(flip_gradient(self.tf_feature_embedding))
#
#         self.tf_mask = tf.sign(self.tf_label + 1)
#         self.tf_regression_loss = tf.reduce_mean(tf.square(self.tf_mask * (self.tf_label - self.tf_label_prediction)))
#         # self.tf_regression_loss = tf.losses.mean_squared_error(self.tf_label, self.tf_label_prediction)
#         self.tf_discrimination_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
#             labels=self.tf_input_domain_feature, logits=self.tf_domain_prediction))
#         self.balanced_loss = lbda * self.tf_regression_loss + (1 - lbda) * self.tf_discrimination_loss
#         self.regularized_loss = (1 - alpha) * self.balanced_loss + alpha * self.regularizer
#         self.tf_optimize = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.regularized_loss)
#
#         config = tf.ConfigProto()
#         config.gpu_options.allow_growth = True
#         self.sess = tf.Session(config=config)
#         self.sess.run(tf.global_variables_initializer())
#         self.dataset = None
#
#     def fit(self, X, D, Y, vX, vD, vY, epoch=2000):
#         """label matrix Y should have elements larger than 0. An element with -1 value is non-labeled entry"""
#         self.dataset = BatchGenerator(X, D, Y)
#         train_regression_losses = []
#         train_discrimination_losses = []
#         validation_regression_losses = []
#         validation_predictions = []
#         while self.dataset.epoch < epoch:
#             if self.dataset.epoch_changed:
#                 if self.dataset.epoch % 10 == 0:
#                     print("epoch", self.dataset.epoch, end=' ')
#                 rl, dl, re = self.sess.run((self.tf_regression_loss, self.tf_discrimination_loss, self.regularizer),
#                                            feed_dict={self.tf_input_feaure: X,
#                                                       self.tf_input_domain_feature: D,
#                                                       self.tf_label: Y})
#                 train_regression_losses.append(rl)
#                 train_discrimination_losses.append(dl)
#                 if self.dataset.epoch % 10 == 0:
#                     print(dl, np.sqrt(re), np.sqrt(rl), end=' ')
#                 rl, pr = self.sess.run((self.tf_regression_loss, self.tf_label_prediction),
#                                        feed_dict={self.tf_input_feaure: vX,
#                                                   self.tf_input_domain_feature: vD,
#                                                   self.tf_label: vY})
#                 validation_regression_losses.append(rl)
#                 validation_predictions = pr
#                 if self.dataset.epoch % 10 == 0:
#                     print(np.sqrt(rl))
#             x, d, y = self.dataset.get_batch()
#             self.sess.run(self.tf_optimize, feed_dict={self.tf_input_feaure: x,
#                                                        self.tf_input_domain_feature: d,
#                                                        self.tf_label: y})
#         # plt.figure("train regression MSE")
#         # plt.plot(train_regression_losses)
#         # plt.figure("train discrimination CE")
#         # plt.plot(train_discrimination_losses)
#         # plt.figure("validation regression MSE")
#         # plt.plot(validation_regression_losses)
#         # plt.figure("validation discrimination CE")
#         # plt.plot(validation_discrimination_losses)
#         return train_regression_losses, \
#                train_discrimination_losses, \
#                validation_regression_losses, \
#                validation_predictions
#
#
# class TResNetRegressPC:
#     def __init__(self, feature_size, domain_size, label_size=1, lbda=0.5, lr=1e-5, alpha=0, lN=2):
#         self.feature_size = feature_size
#         self.domain_size = domain_size
#         self.label_size = label_size
#
#         self.fe = FeatureExtractorPC(feature_size=feature_size,
#                                      domain_size=domain_size,
#                                      node_nums=(80, 40), lN=lN)
#         self.dd = Discriminator(feature_size=40, node_nums=(20, 20, domain_size))
#         self.rg = Regressor(feature_size=40, node_nums=(20, 10, label_size))
#         self.tf_input_feaure = tf.placeholder(dtype=tf.float32, shape=(None, feature_size))
#         self.tf_input_domain_feature = tf.placeholder(dtype=tf.float32, shape=(None, domain_size))
#         self.tf_domain_index = tf.argmax(self.tf_input_domain_feature, axis=1)
#         self.tf_label = tf.placeholder(dtype=tf.float32, shape=(None, label_size))
#         self.tf_feature_embedding, self.regularizer = self.fe(self.tf_input_feaure, self.tf_domain_index)
#         self.tf_label_prediction = self.rg(self.tf_feature_embedding)
#         self.tf_domain_prediction = self.dd(flip_gradient(self.tf_feature_embedding))
#
#         self.tf_mask = tf.sign(self.tf_label + 1)
#         self.tf_regression_loss = tf.reduce_mean(tf.square(self.tf_mask * (self.tf_label - self.tf_label_prediction)))
#         # self.tf_regression_loss = tf.losses.mean_squared_error(self.tf_label, self.tf_label_prediction)
#         self.tf_discrimination_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
#             labels=self.tf_input_domain_feature, logits=self.tf_domain_prediction))
#         self.balanced_loss = lbda * self.tf_regression_loss + (1 - lbda) * self.tf_discrimination_loss
#         self.regularized_loss = (1 - alpha) * self.balanced_loss + alpha * self.regularizer
#         self.tf_optimize = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.regularized_loss)
#
#         config = tf.ConfigProto()
#         config.gpu_options.allow_growth = True
#         self.sess = tf.Session(config=config)
#         self.sess.run(tf.global_variables_initializer())
#         self.dataset = None
#
#     def fit(self, X, D, Y, vX, vD, vY, epoch=2000):
#         """label matrix Y should have elements larger than 0. An element with -1 value is non-labeled entry"""
#         self.dataset = BatchGenerator(X, D, Y)
#         train_regression_losses = []
#         train_discrimination_losses = []
#         validation_regression_losses = []
#         validation_predictions = []
#         while self.dataset.epoch < epoch:
#             if self.dataset.epoch_changed:
#                 if self.dataset.epoch % 10 == 0:
#                     print("epoch", self.dataset.epoch, end=' ')
#                 rl, dl, re = self.sess.run((self.tf_regression_loss, self.tf_discrimination_loss, self.regularizer),
#                                            feed_dict={self.tf_input_feaure: X,
#                                                       self.tf_input_domain_feature: D,
#                                                       self.tf_label: Y})
#                 train_regression_losses.append(rl)
#                 train_discrimination_losses.append(dl)
#                 if self.dataset.epoch % 10 == 0:
#                     print(dl, np.sqrt(re), np.sqrt(rl), end=' ')
#                 rl, pr = self.sess.run((self.tf_regression_loss, self.tf_label_prediction),
#                                        feed_dict={self.tf_input_feaure: vX,
#                                                   self.tf_input_domain_feature: vD,
#                                                   self.tf_label: vY})
#                 validation_regression_losses.append(rl)
#                 validation_predictions = pr
#                 if self.dataset.epoch % 10 == 0:
#                     print(np.sqrt(rl))
#             x, d, y = self.dataset.get_batch()
#             self.sess.run(self.tf_optimize, feed_dict={self.tf_input_feaure: x,
#                                                        self.tf_input_domain_feature: d,
#                                                        self.tf_label: y})
#         # plt.figure("train regression MSE")
#         # plt.plot(train_regression_losses)
#         # plt.figure("train discrimination CE")
#         # plt.plot(train_discrimination_losses)
#         # plt.figure("validation regression MSE")
#         # plt.plot(validation_regression_losses)
#         # plt.figure("validation discrimination CE")
#         # plt.plot(validation_discrimination_losses)
#         return train_regression_losses, \
#                train_discrimination_losses, \
#                validation_regression_losses, \
#                validation_predictions
