"""SK-learn grid-search and test scripts for Multi Layer Perceptron models"""

__author__ = "Gabriel Urbain"
__copyright__ = "Copyright 2017, Gabriel Urbain"

__license__ = "MIT"
__version__ = "0.2"
__maintainer__ = "Gabriel Urbain"
__email__ = "gabriel.urbain@ugent.be"
__status__ = "Research"
__date__ = "September 1st, 2017"


import create_submission
import preprocessing
import linear
import utils

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier, MLPRegressor

BATCH_SIZE = 1000
N_PROCESS = 10


def test(predict=False, ft_name=preprocessing.DEFAULT_FT_LOCATION + "/ft_max_scaler.pkl"):

    # 1. Import the features and target
    print "1. Import the features and target\n"
    test_ft, ft, target = preprocessing.load_ft(ft_name)

    training_ft, validate_ft, training_target, validate_target = \
        train_test_split(ft, target, test_size=preprocessing.VALIDATE_PART, random_state=preprocessing.SEED)
    if not predict:
        print "Training features size: " + str(training_ft.shape) +\
              " and validation features size: " + str(validate_ft.shape)
        print "Training target size: " + str(len(training_target)) + \
              " and validation target size: " + str(len(validate_target)) + "\n"

    # 2. Create the NN
    print "2. Create the Neural Network"
    batch_size = BATCH_SIZE
    layers = (50,)
    activation = "tanh"
    alpha = 1e-4
    learning_rate_init = 0.01
    learning_rate = "adaptive"
    problem = "classification"
    if problem == "classification":
        clf = MLPClassifier(alpha=alpha, activation=activation, learning_rate=learning_rate,
                            batch_size=batch_size, learning_rate_init=learning_rate_init,
                            hidden_layer_sizes=layers, warm_start=True)
    else:
        clf = MLPRegressor(alpha=alpha, activation=activation, learning_rate=learning_rate,
                           batch_size=batch_size, learning_rate_init=learning_rate_init,
                           hidden_layer_sizes=layers, warm_start=True)
        clf.loss = "kaggle_compet_loss"
    lab = preprocessing.Float2Labels()
    classes = np.unique(training_target)
    s = "\tProblem: " + str(problem)
    s += "\n\tBatch size: " + str(batch_size)
    s += "\n\tRegularization parameter: " + str(alpha)
    s += "\n\tNetwork layers size: " + str(layers)
    s += "\n\tNeuron activation function: " + str(activation)
    s += "\n\tLearning method: " + str(learning_rate) + " with initialization to " + str(learning_rate_init)
    s += "\n\tClasses: " + str(classes) + "\n"
    print s

    if not predict:
        # 3. Train the NN
        print "3. Train the Neural Network"
        print "Expected number of iterations: " + str(training_ft.shape[0] / batch_size)
        training_score = []
        validate_score = []
        validate_pred = None

        for i in xrange(0, training_ft.shape[0]/batch_size):
            batch_ft = training_ft[i*batch_size:(i+1)*batch_size]
            batch_target = training_target[i*batch_size:(i+1)*batch_size]

            if i == 0:
                clf = clf.partial_fit(batch_ft, batch_target, classes)
            else:
                clf = clf.partial_fit(batch_ft, batch_target)

            training_pred = lab.transform(clf.predict(training_ft))
            validate_pred = lab.transform(clf.predict(validate_ft))

            training_score.append(preprocessing.loss_fct(training_pred, training_target))
            validate_score.append(preprocessing.loss_fct(validate_pred, validate_target))

            print "Iteration %i Score on training set: %0.3f and validation set: %0.3f" % \
                  (i, training_score[-1], validate_score[-1])

        # 4. Plot
        print "\n4. Plot and save the training and validation loss functions\n"
        fig, ax1 = plt.subplots()
        ax1.plot(training_score, color=utils.get_style_colors()[0], label="Training")
        ax1.plot(validate_score, color=utils.get_style_colors()[1], label="Validation")
        plt.legend()
        ax1.set_ylabel('Score')
        ax1.set_xlabel('Batch Number')
        fig.tight_layout()
        filename = linear.DEFAULT_TRAIN_LOCATION + "/MLP/cv_mlp_" + str(utils.timestamp())
        utils.dump_pickle((training_score, validate_score), filename + ".pkl")
        text_file = open(filename + ".txt", "w")
        text_file.write(s)
        text_file.close()
        plt.savefig(filename + ".png", format='png', dpi=300)
        plt.close()
        cnf_matrix = confusion_matrix(validate_pred, validate_target)
        utils.plot_confusion_matrix(cnf_matrix, classes=classes, normalize=True, filename=filename + "_conf_mat")

    if predict:
        # 3. Train on full dataset
        print "3. Train\n"
        print "Expected number of iterations: " + str(ft.shape[0] / batch_size)
        training_score = []

        for i in xrange(0, ft.shape[0] / batch_size):
            batch_ft = ft[i * batch_size:(i + 1) * batch_size]
            batch_target = target[i * batch_size:(i + 1) * batch_size]

            if i == 0:
                clf = clf.partial_fit(batch_ft, batch_target, classes)
            else:
                clf = clf.partial_fit(batch_ft, batch_target)

            training_pred = lab.transform(clf.predict(ft))
            training_score.append(preprocessing.loss_fct(training_pred, target))

            print "Iteration %i Score on training set: %0.3f" % (i, training_score[-1])

        # 4. Predict test results
        print "4. Predict and save test results"
        test_pred = lab.transform(clf.predict(test_ft))
        training_pred = lab.transform(clf.predict(ft))
        print "Score on training set: %0.5f" % preprocessing.loss_fct(training_pred, target)

        filename = linear.DEFAULT_PRED_LOCATION + "/mlp_" + str(utils.timestamp())
        utils.dump_pickle(test_pred, filename + ".pkl")
        create_submission.write_predictions_to_csv(test_pred, filename + ".csv")


def grid_search():
    # 1. Import the features and target
    print "1. Import the features and target\n"
    test_features, feature, target = preprocessing.load_ft(
        preprocessing.DEFAULT_FT_LOCATION + "/ft_max_scaler.pkl")
    training_ft, validate_ft, training_target, validate_target = \
        train_test_split(feature, target, test_size=preprocessing.VALIDATE_PART,
                         random_state=preprocessing.SEED)
    print "Training features size: " + str(training_ft.shape) + \
          " and validation features size: " + str(validate_ft.shape)
    print "Training target size: " + str(len(training_target)) + \
          " and validation target size: " + str(len(validate_target)) + "\n"

    # 2. Create the NN
    print "2. Create the Neural Network"
    batch_size = BATCH_SIZE
    layers = (50,)
    activation = "tanh"
    alpha = 1e-4
    learning_rate_init = 0.01
    learning_rate = "adaptive"
    problem = "classification"
    if problem == "classification":
        clf = MLPClassifier(alpha=alpha, activation=activation, learning_rate=learning_rate,
                            batch_size=batch_size, learning_rate_init=learning_rate_init,
                            hidden_layer_sizes=layers, warm_start=True)
    else:
        clf = MLPRegressor(alpha=alpha, activation=activation, learning_rate=learning_rate,
                           batch_size=batch_size, learning_rate_init=learning_rate_init,
                           hidden_layer_sizes=layers, warm_start=True)
        clf.loss = "kaggle_compet_loss"

    classes = np.unique(training_target)
    s = "\tProblem: " + str(problem)
    s += "\n\tBatch size: " + str(batch_size)
    s += "\n\tRegularization parameter: " + str(alpha)
    s += "\n\tNetwork layers size: " + str(layers)
    s += "\n\tNeuron activation function: " + str(activation)
    s += "\n\tLearning method: " + str(learning_rate) + " with initialization to " + str(learning_rate_init)
    s += "\n\tClasses: " + str(classes) + "\n"
    print s
    parameters = {'alpha': np.logspace(-7, -2, num=6).tolist(),
                  'hidden_layer_sizes': [(i,) for i in [10, 15, 20, 50, 100]],
                  'batch_size': [1000]}

    # 3. Create and run the cross-validation search method
    print "3. Perform grid search"
    print "    Parameters: ",
    pprint(parameters)
    print ""

    loss = make_scorer(preprocessing.loss_fct, greater_is_better=False)
    gs = GridSearchCV(clf, parameters, n_jobs=N_PROCESS, verbose=2, scoring=loss)
    gs.fit(training_ft, training_target)

    # 4. Save the results
    print "4. Save and plot scores"
    print "\nBest score: %0.3f" % -gs.best_score_
    print "Best parameters set:"
    best_parameters = gs.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    r = gs.cv_results_
    utils.dump_pickle(r, linear.DEFAULT_TRAIN_LOCATION + "/cv_mlp_" + utils.timestamp() + ".pkl")


def plot(filename):

    # 1. Load the results
    r = utils.load_pickle(filename)
    alpha = r['param_alpha']
    layers = r['param_hidden_layer_sizes']
    batch_size = r['param_batch_size']
    sv = -r['mean_test_score']

    alpha_r = np.unique(alpha)
    layers_r = np.unique(layers)
    batch_size_r = np.unique(batch_size)

    sv = sv.reshape(len(batch_size_r), len(alpha_r), len(layers_r))
    alpha = alpha.reshape(len(batch_size_r), len(alpha_r), len(layers_r))
    layers = layers.reshape(len(batch_size_r), len(alpha_r), len(layers_r))

    for i in xrange(len(batch_size_r)):
        s = sv[i, :, :]
        a = alpha[i, :, :]
        l = layers[i, :, :]
        print s
        print a
        print l
        idx = s < np.sort(s.flatten())[5]
        best_s = s[idx]
        best_a = a[idx]
        best_l = l[idx]
        print "Best results (score, alpha, layer_units): ",
        for i in range(best_s.size):
            print "(%0.3f, " % best_s[-i] + str(best_a[-i]) + ", " + str(best_l[-i]) + ")  ",
        mi = np.min(s)
        ma = np.max(s)
        plt.imshow(s, interpolation='nearest', cmap=plt.cm.hot_r,
                   norm=utils.MidpointNormalize(vmin=mi, midpoint=mi+(ma-mi)/2))
        plt.xlabel('First layer units')
        plt.ylabel('Regularization parameter alpha')
        plt.colorbar()
        plt.yticks(np.arange(len(alpha_r)), alpha_r, rotation=45)
        plt.xticks(np.arange(len(layers_r)), layers_r)
        plt.title('Validation score')
        plt.tight_layout()
        filename = linear.DEFAULT_TRAIN_LOCATION + "/cv_mlp_hotmap.png"
        plt.savefig(filename, format='png', dpi=300)
        plt.close()


def vote(predict=True, add_linear=True, ft_name=preprocessing.DEFAULT_FT_LOCATION + "/ft_max_scaler.pkl"):

    # 1. Import the features and target
    print "1. Import the features and target\n"
    test_ft, ft, target = preprocessing.load_ft(ft_name)

    training_ft, validate_ft, training_target, validate_target = \
        train_test_split(ft, target, test_size=preprocessing.VALIDATE_PART, random_state=preprocessing.SEED)
    if not predict:
        print "Training features size: " + str(training_ft.shape) + \
              " and validation features size: " + str(validate_ft.shape)
        print "Training target size: " + str(len(training_target)) + \
              " and validation target size: " + str(len(validate_target)) + "\n"

    # 2. Create the NN
    print "2. Create the Neural Networks"
    batch_size = BATCH_SIZE
    clf1 = MLPClassifier(alpha=0.0001, activation="tanh", learning_rate="adaptive",
                         batch_size=batch_size, learning_rate_init=0.01,
                         hidden_layer_sizes=(50,), warm_start=True)
    clf2 = MLPClassifier(alpha=0.0001, activation="tanh", learning_rate="adaptive",
                         batch_size=batch_size, learning_rate_init=0.01,
                         hidden_layer_sizes=(10,), warm_start=True)
    clf3 = MLPClassifier(alpha=1e-07, activation="tanh", learning_rate="adaptive",
                         batch_size=batch_size, learning_rate_init=0.01,
                         hidden_layer_sizes=(20,), warm_start=True)
    clf4 = MLPClassifier(alpha=1e-07, activation="tanh", learning_rate="adaptive",
                         batch_size=batch_size, learning_rate_init=0.01,
                         hidden_layer_sizes=(10,), warm_start=True)
    clf5 = MLPClassifier(alpha=1e-07, activation="tanh", learning_rate="adaptive",
                         batch_size=batch_size, learning_rate_init=0.01,
                         hidden_layer_sizes=(15,), warm_start=True)
    classes = np.unique(training_target)
    print "\tClasses: " + str(classes) + "\n"

    if not predict:
        # 3. Train the NN
        print "3. Train the Neural Network"
        print "Expected number of iterations: " + str(training_ft.shape[0] / batch_size)
        training_score = []
        validate_score = []

        for i in xrange(0, training_ft.shape[0] / batch_size):
            batch_ft = training_ft[i * batch_size:(i + 1) * batch_size]
            batch_target = training_target[i * batch_size:(i + 1) * batch_size]

            if i == 0:
                clf1 = clf1.partial_fit(batch_ft, batch_target, classes)
                clf2 = clf2.partial_fit(batch_ft, batch_target, classes)
                clf3 = clf3.partial_fit(batch_ft, batch_target, classes)
                clf4 = clf4.partial_fit(batch_ft, batch_target, classes)
                clf5 = clf5.partial_fit(batch_ft, batch_target, classes)
            else:
                clf1 = clf1.partial_fit(batch_ft, batch_target)
                clf2 = clf2.partial_fit(batch_ft, batch_target)
                clf3 = clf3.partial_fit(batch_ft, batch_target)
                clf4 = clf4.partial_fit(batch_ft, batch_target)
                clf5 = clf5.partial_fit(batch_ft, batch_target)

            tp1 = preprocessing.loss_fct(clf1.predict(training_ft), training_target)
            vp1 = preprocessing.loss_fct(clf1.predict(validate_ft), validate_target)
            tp2 = preprocessing.loss_fct(clf2.predict(training_ft), training_target)
            vp2 = preprocessing.loss_fct(clf2.predict(validate_ft), validate_target)
            tp3 = preprocessing.loss_fct(clf3.predict(training_ft), training_target)
            vp3 = preprocessing.loss_fct(clf3.predict(validate_ft), validate_target)
            tp4 = preprocessing.loss_fct(clf4.predict(training_ft), training_target)
            vp4 = preprocessing.loss_fct(clf4.predict(validate_ft), validate_target)
            tp5 = preprocessing.loss_fct(clf5.predict(training_ft), training_target)
            vp5 = preprocessing.loss_fct(clf5.predict(validate_ft), validate_target)

            # HACK: custom voting classifier because partial_fit not implemented
            probt1 = clf1.predict_proba(training_ft)
            probt2 = clf2.predict_proba(training_ft)
            probt3 = clf3.predict_proba(training_ft)
            probt4 = clf4.predict_proba(training_ft)
            probt5 = clf5.predict_proba(training_ft)
            eprobt = np.argmax((probt1 + probt2 + probt3 + probt4 + probt5) / 5, axis=1) + 1
            ept = preprocessing.loss_fct(eprobt, training_target)
            probv1 = clf1.predict_proba(validate_ft)
            probv2 = clf2.predict_proba(validate_ft)
            probv3 = clf3.predict_proba(validate_ft)
            probv4 = clf4.predict_proba(validate_ft)
            probv5 = clf5.predict_proba(validate_ft)
            eprobv = np.argmax((probv1 + probv2 + probv3 + probv4 + probv5) / 5, axis=1) + 1
            epv = preprocessing.loss_fct(eprobv, validate_target)

            training_score.append(ept)
            validate_score.append(epv)

            print "Iteration %i:\tCLF training scores: [%0.3f %0.3f %0.3f %0.3f %0.3f] and ENS training score: %0.3f" \
                  % (i, tp1, tp2, tp3, tp4, tp5, training_score[-1])
            print "\t\tCLF validate scores: [%0.3f %0.3f %0.3f %0.3f %0.3f] and ENS validate score: %0.3f\n" \
                  % (vp1, vp2, vp3, vp4, vp5, validate_score[-1])

        # 4. Plot
        print "\n4. Plot and save the training and validation loss functions\n"
        fig, ax1 = plt.subplots()
        ax1.plot(training_score, color=utils.get_style_colors()[0])
        ax1.plot(validate_score, color=utils.get_style_colors()[1])
        ax1.set_ylabel('Score')
        fig.tight_layout()
        filename = linear.DEFAULT_TRAIN_LOCATION + "/MLP/cv_mlp_ensemble" + str(utils.timestamp())
        utils.dump_pickle((training_score, validate_score), filename + ".pkl")
        text_file = open(filename + ".txt", "w")
        text_file.write("Ensemble function... See commit version for values")
        text_file.close()
        plt.savefig(filename + ".png", format='png', dpi=300)
        plt.close()

    if predict:
        # 3. Train on full dataset
        print "3. Train\n"
        print "Expected number of iterations: " + str(ft.shape[0] / batch_size)
        training_score = []

        for i in xrange(0, ft.shape[0] / batch_size):
            batch_ft = ft[i * batch_size:(i + 1) * batch_size]
            batch_target = target[i * batch_size:(i + 1) * batch_size]

            if i == 0:
                clf1 = clf1.partial_fit(batch_ft, batch_target, classes)
                clf2 = clf2.partial_fit(batch_ft, batch_target, classes)
                clf3 = clf3.partial_fit(batch_ft, batch_target, classes)
                clf4 = clf4.partial_fit(batch_ft, batch_target, classes)
                clf5 = clf5.partial_fit(batch_ft, batch_target, classes)
            else:
                clf1 = clf1.partial_fit(batch_ft, batch_target)
                clf2 = clf2.partial_fit(batch_ft, batch_target)
                clf3 = clf3.partial_fit(batch_ft, batch_target)
                clf4 = clf4.partial_fit(batch_ft, batch_target)
                clf5 = clf5.partial_fit(batch_ft, batch_target)

            tp1 = preprocessing.loss_fct(clf1.predict(ft), target)
            tp2 = preprocessing.loss_fct(clf2.predict(ft), target)
            tp3 = preprocessing.loss_fct(clf3.predict(ft), target)
            tp4 = preprocessing.loss_fct(clf4.predict(ft), target)
            tp5 = preprocessing.loss_fct(clf5.predict(ft), target)

            # HACK: custom voting classifier because partial_fit not implemented
            probt1 = clf1.predict_proba(ft)
            probt2 = clf2.predict_proba(ft)
            probt3 = clf3.predict_proba(ft)
            probt4 = clf4.predict_proba(ft)
            probt5 = clf5.predict_proba(ft)
            eprobt = np.argmax((probt1 + probt2 + probt3 + probt4 + probt5) / 5, axis=1) + 1
            ept = preprocessing.loss_fct(eprobt, target)
            training_score.append(ept)

            print "Iteration %i:\tCLF training scores: [%0.3f %0.3f %0.3f %0.3f %0.3f] and ENS training score: %0.3f" \
                  % (i, tp1, tp2, tp3, tp4, tp5, training_score[-1])

        # 5. Predict test results
        print "\n5. Predict and save test results"
        # Hack again
        testp1 = clf1.predict_proba(test_ft)
        testp2 = clf2.predict_proba(test_ft)
        testp3 = clf3.predict_proba(test_ft)
        testp4 = clf4.predict_proba(test_ft)
        testp5 = clf5.predict_proba(test_ft)
        test_pred = np.argmax((testp1 + testp2 + testp3 + testp4 + testp5) / 5, axis=1) + 1
        trainp1 = clf1.predict_proba(ft)
        trainp2 = clf2.predict_proba(ft)
        trainp3 = clf3.predict_proba(ft)
        trainp4 = clf4.predict_proba(ft)
        trainp5 = clf5.predict_proba(ft)
        train_pred = np.argmax((trainp1 + trainp2 + trainp3 + trainp4 + trainp5) / 5, axis=1) + 1

        print "Score on training set: %0.5f\n" % preprocessing.loss_fct(train_pred, target)

        filename = linear.DEFAULT_PRED_LOCATION + "/mlp_ensemble_" + str(utils.timestamp())
        utils.dump_pickle(test_pred, filename + ".pkl")
        create_submission.write_predictions_to_csv(test_pred, filename + ".csv")

        if add_linear:
            # 6. Mix linear and non-linear probabilities
            print "6. Predict an ensemble with linear classification and save test results"
            lclf = LogisticRegression(verbose=0, C=1.3, penalty='l1')
            lclf.fit(ft, target)

            testp1 = clf1.predict_proba(test_ft)
            testp2 = clf2.predict_proba(test_ft)
            testp3 = clf3.predict_proba(test_ft)
            testp4 = clf4.predict_proba(test_ft)
            testp5 = clf5.predict_proba(test_ft)
            testpl = lclf.predict_proba(test_ft)
            test_pred = np.argmax((0.3*testp1 + 0.27*testp2 + 0.24*testp3 +
                                   0.21*testp4 + 0.18*testp5 + testpl) / 2.2, axis=1) + 1
            trainp1 = clf1.predict_proba(ft)
            trainp2 = clf2.predict_proba(ft)
            trainp3 = clf3.predict_proba(ft)
            trainp4 = clf4.predict_proba(ft)
            trainp5 = clf5.predict_proba(ft)
            trainpl = lclf.predict_proba(ft)
            train_pred = np.argmax((0.3*trainp1 + 0.27*trainp2 + 0.24*trainp3 +
                                    0.21*trainp4 + 0.18*trainp5 + trainpl) / 2.2, axis=1) + 1

            print "Score on training set: %0.5f" % preprocessing.loss_fct(train_pred, target)
            filename = linear.DEFAULT_PRED_LOCATION + "/mlp_ensemble_with_linear_" + str(utils.timestamp())
            utils.dump_pickle(test_pred, filename + ".pkl")
            create_submission.write_predictions_to_csv(test_pred, filename + ".csv")


if __name__ == '__main__':

    args = sys.argv

    if args[1] == "test":
        test()
    elif args[1] == "predict":
        test(predict=True)
    elif args[1] == "gs":
        grid_search()
    elif args[1] == "plot":
        plot(args[2])
    elif args[1] == "vote":
        vote()
    else:
        print "Option does not exist. Please, check the feature_selection.py file"
