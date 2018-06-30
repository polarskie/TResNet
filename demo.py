import numpy as np
from sklearn.svm import SVC
import sys
sys.path.insert(0, '../../')
import TResNetClass

n_domain = 10
n_feature = 3
sample_per_dom = 200
X_list = []
Y_list = []
D_list = []
mean_c1 = np.ones((n_feature,))
var_c1 = np.eye(n_feature)
mean_c2 = -np.ones((n_feature,))
var_c2 = np.eye(n_feature)
dom_discrepancy_var = 1.0

for dom_i in range(n_domain):
    dom_discrepancy = np.random.randn(n_feature) * dom_discrepancy_var
    x_c1 = np.random.multivariate_normal(mean_c1 + dom_discrepancy, var_c1, size=int(sample_per_dom / 2))
    x_c2 = np.random.multivariate_normal(mean_c2 + dom_discrepancy, var_c2, size=int(sample_per_dom / 2))
    X_list.append(np.concatenate((x_c1, x_c2), axis=0))
    Y_list.append(np.concatenate(
        (np.concatenate((np.ones((int(sample_per_dom / 2), 1)), np.zeros((int(sample_per_dom / 2), 1))), axis=1),
         np.concatenate(
             (np.zeros((int(sample_per_dom / 2), 1)), np.ones((int(sample_per_dom / 2), 1))),
             axis=1)), axis=0))
    D_list.append(np.concatenate((np.zeros((int(sample_per_dom / 2), dom_i)), np.ones((int(sample_per_dom / 2), 1)),
                             np.zeros((int(sample_per_dom / 2), n_domain - dom_i - 1))), axis=1))

# test baseline
baseline_accs = []
for target_dom_i in range(n_domain):
    X_train = np.concatenate([X for i, X in enumerate(X_list) if i != target_dom_i])
    Y_train = np.argmax(np.concatenate([Y for i, Y in enumerate(Y_list) if i != target_dom_i]), axis=1)
    X_test = X_list[target_dom_i]
    Y_test = np.argmax(Y_list[target_dom_i], axis=1)
    svc = SVC()
    svc.fit(X_train, Y_train)
    baseline_accs.append(np.mean([1.0 if y_true == y_predict else 0.0 for y_true, y_predict in zip(Y_test, svc.predict(X_test))]))
print(np.mean(baseline_accs))
print(baseline_accs)

# run DANN
dann_accs = []
for target_dom_i in range(n_domain):
    X_train = np.concatenate([X for X in X_list])
    Y_train = np.concatenate([Y if i != target_dom_i else np.zeros(Y_list[i].shape) for i, Y in enumerate(Y_list)])
    D_train = np.concatenate([np.concatenate((np.zeros((sample_per_dom, i)),
                                              np.ones((sample_per_dom, 1)),
                                              np.zeros((sample_per_dom, n_domain - i - 1))), axis=1) for i in range(n_domain)], axis=0)
    X_test = X_list[target_dom_i]
    Y_test = Y_list[target_dom_i]
    D_test = np.concatenate((np.zeros((sample_per_dom, target_dom_i)),
                             np.ones((sample_per_dom, 1)),
                             np.zeros((sample_per_dom, n_domain - target_dom_i - 1))), axis=1)

    # X_train = np.concatenate([X if i != domain_i else
    #                           np.repeat(X, repeats=14, axis=0) for i, X in enumerate(Xs)])
    # D_train = np.concatenate([np.zeros(shape=[X.shape[0], 2]) + [1.0, 0.0] if i != domain_i else
    #                           np.zeros(shape=[X.shape[0] * 14, 2]) + [0.0, 1.0] for i, X in enumerate(Xs)])
    # Y_train = np.concatenate([Y if i != domain_i else
    #                           np.zeros(shape=(Y.shape[0] * 14, Y.shape[1]), dtype=float)
    #                           for i, Y in enumerate(Ys)])
    # X_test = Xs[domain_i]
    # D_test = np.zeros(shape=[X_test.shape[0], 2]) + [0.0, 1.0]
    # # D_test = Ds[domain_i]
    # Y_test = Ys[domain_i]
    # X_train, X_test = normalize(X_train, X_test)
    arg_dict = {'X': X_train,
                'D': D_train,
                'Y': Y_train,
                'tX': X_test,
                'tD': D_test,
                'tY': Y_test,
                'feature_size': n_feature,
                'domain_size': n_domain,
                'label_size': 2,
                'id': target_dom_i,
                'printlog': True,
                'display': True,
                'lr': 0.001,
                'alpha': 0.5,
                'beta': 0.0,
                'max_flip_ratio': 1.0,
                'fea_ext_layers': [n_feature, ],
                'dom_dis_layers': [n_domain, ],
                'lab_pre_layers': [2, ],
                'residual_type': '',
                'batch_size': 80,
                'info': "baseline acc: " + str(baseline_accs[target_dom_i])}
    TResNetClass.tres_classfiy(arg_dict)
    quit()