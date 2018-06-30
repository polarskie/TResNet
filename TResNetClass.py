from TResNet import *
import time
import sys
import os


class TResNetClassify:
    def __init__(self,
                 feature_size,
                 domain_size,
                 label_size=2,
                 alpha=0.5,
                 beta=0,
                 residual_type='b',
                 lr=1e-5,
                 max_flip_ratio=0.01,
                 lN=2,
                 fea_ext_layers=(80, 40, 20,),
                 dom_dis_layers=None,
                 lab_pre_layers=None,
                 activation_function='leaky_relu',
                 info='',
                 id=-1,
                 display=False,
                 printlog=False,
                 GPU=1):
        ################################################################################################################
        # feature_size: dimension of the input
        # domain_size: number of the domains
        # label_size: number of the classes
        # alpha: a trade off hyper-parameter. see the next one for detail
        # beta: a trade off hyper-parameter.
        #       the loss function of the network is
        #           (1 - beta)((1 - alpha) * label_prediction_loss + alpha * domain_prediction_loss) + beta * regularization_term
        # residual_type: a str containing or not 'w' and 'b'. if it contains 'w', different domains will have feature
        #                extractors with different W. i.e., W(k) + res_W1(k) for domain 1 in layer k, W(k) + res_W2(k)
        #                for domain 2 in layer k, etc. if it contains 'b', different domains will have feature
        #                extractors with different bias b. i.e., b(k) + res_b1(k) for domain 1 in layer k, b(k) + res_b2(k)
        #                for domain 2 in layer k, etc. if it is '', the network becomes DANN (Domain-Adversarial training
        #                of Neural Network)
        # lr: learning rate of the optimizer
        # max_flip_ratio: the factor for the flip_gradient module
        # lN: type of norms for the regularization
        # fea_ext_layers: shape of the feature extractor. the Ws for the feature extractor will be in the shape of
        #                 [feature_size, fea_ext_layers[0]], [fea_ext_layers[0], fea_ext_layers[1]], [fea_ext_layers[1],
        #                 fea_ext_layers[2]], etc
        # dom_dis_layers: shape of the domain discriminator. see fea_ext_layers for details
        # lab_pre_layers: shape of the label predictor. see fea_ext_layers for details
        # activation_function: can be one of "sigmoid", "relu", "elu", and "leaky_relu". the default is "identity"
        # info: any str that would be printed during the training
        # id: identifier of this instance
        # display: whether or not show the training status with graphs and plots (time consuming)
        # printlog: whether or not print the training status to the terminal
        # GPU: which gpu to use. for single gpu, the default '1' should be fine
        ################################################################################################################
        os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)
        self.feature_size = feature_size
        self.domain_size = domain_size
        self.label_size = label_size
        self.max_flip_ratio = max_flip_ratio
        self.last_dl = None
        self.id = id
        self.active = activation_function
        self.alpha = alpha
        self.info = info
        self.display = display
        self.printlog = printlog

        self.fe = FeatureExtractor(feature_size=feature_size,
                                   domain_size=domain_size,
                                   node_nums=fea_ext_layers,
                                   lN=lN,
                                   activation=self.active,
                                   residual_type=residual_type)
        self.dd = Discriminator(feature_size=fea_ext_layers[-1] if len(fea_ext_layers) > 0 else feature_size,
                                node_nums=(domain_size,) if dom_dis_layers is None else dom_dis_layers,
                                activation=self.active)
        self.lp = Discriminator(feature_size=fea_ext_layers[-1] if len(fea_ext_layers) > 0 else feature_size,
                                node_nums=(label_size,) if lab_pre_layers is None else lab_pre_layers,
                                activation=self.active)

        self.tf_input_feaure = tf.placeholder(dtype=tf.float32, shape=(None, feature_size))
        self.tf_input_domain_feature = tf.placeholder(dtype=tf.float32, shape=(None, domain_size))
        self.tf_label = tf.placeholder(dtype=tf.float32, shape=(None, label_size))

        self.tf_alpha = tf.placeholder(dtype=tf.float32)
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

        self.balanced_loss = (1 - self.tf_alpha) * self.tf_classification_loss + self.tf_alpha * self.tf_discrimination_loss
        self.regularized_loss = (1 - beta) * self.balanced_loss + beta * self.regularizer
        # self.tf_optimize = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(self.regularized_loss)
        self.tf_optimize = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.regularized_loss)
        # tf.train.MomentumOptimizer()

        config = tf.ConfigProto()
        # config = tf.ConfigProto(device_count={'GPU': GPU})
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.dataset = None
        self.dl_buf = []
        self.ad_train_switch = False

    def get_flip_ratio(self):
        return self.max_flip_ratio

    def fit(self, X, D, Y, tX, tD, tY, vX=None, vD=None, vY=None, epoch=5000, monitor_at=1, batch_size=100):
        ################################################################################################################
        # X: the input data for the network in form of numpy array with shape [sample_num, feature_size]. both data
        #    samples for source and target domains should be provided
        # D: the domain information for X, in form of numpy array with shape [sample_num, domain_size]. each row
        #    contains several '0's and one or zero '1'. the domain is indicated by the position of the '1'. e.g., if we
        #    have three data samples. the first one belongs to domain 0, the second one belongs to domain 1, and the
        #    third one belongs to domain 2. the corresponding D could be
        #    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        # Y: the corresponding label for X, in form of numpy array with shape [sample_num, label_size]. each row
        #    contains several '0's and one or zero '1'. the class is indicated by the position of the '1'. for data
        #    samples whose labels are unknown, use all-zero vector. e.g., if we have three data samples. the first one
        #    belongs to class 0, the second one belongs to class 1, and the third one is not labeled. the corresponding
        #    Y could be [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
        # tX: the input data for the network for test
        # tD: the domain information for tX
        # tY: the label information for tX
        # epoch: the number of epochs to train before stop
        # monitor_at: the interval between the evaluations of the network with tX, tD, and tY.
        # batch_size: batch　size
        plt.close("all")
        self.dataset = BatchGenerator(X, D, Y, batch_size=batch_size)
        # train_regression_losses = []
        # train_discrimination_losses = []
        # validation_regression_losses = []
        # validation_predictions = []
        # test_regression_losses = []
        # test_predictions = []
        if self.display:
            fig = plt.figure(figsize=(12, 10), tight_layout=True)
            gs = gridspec.GridSpec(4, 3, height_ratios=(1., 0.7, 0.7, 0.7))
            ax = [fig.add_subplot(gs[0, i]) for i in range(3)]
            ax2 = [fig.add_subplot(gs[i+1, :]) for i in range(3)]
        # fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        # fig2, ax2 = plt.subplots(2, 1, figsize=(15, 5))
        plot_discriminate_loss = []
        plot_discriminate_acc = []
        plot_train_label_prediction_loss = []
        plot_validate_label_prediction_loss = []
        plot_test_label_prediction_loss = []
        plot_test_label_prediction_acc = []
        plot_test_feature_extractor_results = []
        # time_consumption = [0, 0, 0, 0]
        try:
            while self.dataset.epoch < epoch:
                # tmp_time = time.time()
                if self.dataset.epoch_changed and self.dataset.epoch % monitor_at == 0:
                    if self.printlog:
                        print("\n*********************\n", "module", self.id, "epoch", self.dataset.epoch)
                        print(self.info)
                    rl, fe, dl, re, pr = self.sess.run((self.tf_classification_loss, self.tf_feature_embedding,
                                                    self.tf_discrimination_loss, self.regularizer, self.tf_label_prediction),
                                                   feed_dict={self.tf_input_feaure: X,
                                                              self.tf_input_domain_feature: D,
                                                              self.tf_label: Y,
                                                              self.tf_flip_gradient_ratio: 0.1,
                                                              self.tf_alpha: self.alpha})
                    # train_regression_losses.append(rl)
                    # train_discrimination_losses.append(dl)
                    if self.display:
                        plot_discriminate_acc.append(test_domain_discrepancy(fe, D, ax[2]))
                    # print('**********' + str(time.time() - t))
                    if self.printlog:
                        print("discriminate loss:", dl)
                        # print("SVM discriminate acc:", plot_discriminate_acc[-1])
                        print("regularization norm:", np.sqrt(re))
                        print("train label prediction loss:", rl)
                    # if self.dataset.epoch % (10 * monitor_at) == 0:
                    #     plot_test_feature_extractor_results.append(fe)
                    plot_discriminate_loss.append(dl)
                    plot_train_label_prediction_loss.append(rl)
                    if vX is not None:
                        rl, fe, pr = self.sess.run(
                            (self.tf_classification_loss, self.tf_feature_embedding, self.tf_label_prediction),
                            feed_dict={self.tf_input_feaure: vX,
                                       self.tf_input_domain_feature: vD,
                                       self.tf_label: vY,
                                       self.tf_flip_gradient_ratio: 0.1,
                                       self.tf_alpha: self.alpha})

                        acc = np.mean([1. if i == j else 0. for i, j in zip(np.argmax(pr, axis=1),
                                                                            np.argmax(vY, axis=1))])
                        if self.printlog:
                            print("validate label prediction loss:", rl)
                            print("validate label prediction acc:", acc)
                        plot_validate_label_prediction_loss.append(rl)
                    if self.display:
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
                                                      self.tf_alpha: self.alpha})
                    # test_regression_losses.append(rl)
                    # test_predictions = pr
                    acc = np.mean([1. if i == j else 0. for i, j in zip(np.argmax(pr, axis=1),
                                                                        np.argmax(tY, axis=1))])
                    if self.printlog:
                        print("test label prediction loss:", rl)
                        print("test label prediction acc:", acc)
                        print("gradient flip ratio:", self.get_flip_ratio())
                    plot_test_label_prediction_loss.append(rl)
                    plot_test_label_prediction_acc.append(acc)
                    if self.display:
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

                    if plot_train_label_prediction_loss[-1] < 0.1:
                        break

                # time_consumption[0] += (time.time() - tmp_time)
                # tmp_time = time.time()
                x, d, y = self.dataset.get_batch()
                # time_consumption[1] += (time.time() - tmp_time)
                # tmp_time = time.time()

                # time_consumption[2] += (time.time() - tmp_time)
                tmp_time = time.time()
                _ = self.sess.run(self.tf_optimize,
                                  feed_dict={self.tf_input_feaure: x,
                                             self.tf_input_domain_feature: d,
                                             self.tf_label: y,
                                             self.tf_flip_gradient_ratio: self.get_flip_ratio(),
                                             self.tf_alpha: self.alpha})
                # time_consumption[3] += (time.time() - tmp_time)
        except KeyboardInterrupt:
            if self.printlog:
                print('keyboard interruption detected')
        except Exception as e:
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
            if self.printlog:
                print(e)
            ret = {'sample_rate': monitor_at,
                    'discriminate_loss': plot_discriminate_loss,
                    'discriminate_acc': plot_discriminate_acc,
                    'train_label_prediction_loss': plot_train_label_prediction_loss,
                    'validate_label_prediction_loss': plot_validate_label_prediction_loss,
                    'test_label_prediction_loss': plot_test_label_prediction_loss,
                    'test_label_prediction_acc': plot_test_label_prediction_acc,
                    'test_feature_extractor_results': plot_test_feature_extractor_results,
                    'program_corruption': e}
        else:
            ret = {'sample_rate': monitor_at,
                    'discriminate_loss': plot_discriminate_loss,
                    'discriminate_acc': plot_discriminate_acc,
                    'train_label_prediction_loss': plot_train_label_prediction_loss,
                    'validate_label_prediction_loss': plot_validate_label_prediction_loss,
                    'test_label_prediction_loss': plot_test_label_prediction_loss,
                    'test_label_prediction_acc': plot_test_label_prediction_acc,
                    'test_feature_extractor_results': plot_test_feature_extractor_results,
                    'program_corruption': None}
        finally:
            if self.printlog:
                plt.close()
            # plt.ioff()
            # plt.figure('time')
            # plt.bar(np.arange(len(time_consumption)), time_consumption)
            # plt.show()
            return ret


def tres_classfiy(para_dict):
    init_args_name = TResNetClassify.__init__.__code__.co_varnames
    fit_args_name = TResNetClassify.fit.__code__.co_varnames
    init_args_dict = {arg_name: para_dict[arg_name] for arg_name in init_args_name if arg_name in para_dict}
    fit_args_dict = {arg_name: para_dict[arg_name] for arg_name in fit_args_name if arg_name in para_dict}
    clf = TResNetClassify(**init_args_dict)
    return clf.fit(**fit_args_dict)