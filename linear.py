""" Script that transform the content of train in sklearn boolean features """

import preprocessing
import data
import utils

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
import sys
import time

from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer
from sklearn.model_selection import *
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Configuration
plt.style.use('fivethirtyeight')

# Global constants
DEFAULT_TRAIN_LOCATION = 'Train'
DEFAULT_PRED_LOCATION = "Predictions"
N_PROCESS = 4


def grid_search(clf='lr'):

    # Get the data
    print "Load data"
    dataset = data.load_pickled_data()["train"]
    target = preprocessing.create_target(dataset)

    # Create the pipe
    if clf == "lr":
        ft_extractor = preprocessing.create_ft_ct()
        classifier = LogisticRegression(verbose=0, penalty='l1')
        parameters = {'clf__C': np.logspace(-3, 2, num=15).tolist()}
        pipe = Pipeline([('ft', ft_extractor), ('clf', classifier)])
        filename = DEFAULT_TRAIN_LOCATION + "/cv_lr_" + utils.timestamp() + ".pkl"
    elif clf == "lr_all":
        ft_extractor = preprocessing.create_ft_ct_pd_au()
        classifier = LogisticRegression(verbose=0, penalty='l1')
        parameters = {'clf__C': np.logspace(-3, 2, num=15).tolist()}
        pipe = Pipeline([('ft', ft_extractor), ('clf', classifier)])
        filename = DEFAULT_TRAIN_LOCATION + "/cv_lr_all_" + utils.timestamp() + ".pkl"
    elif clf == "lr_all_svd":
        ft_extractor = preprocessing.create_ft_ct()
        ft_reductor = TruncatedSVD()
        classifier = LogisticRegression(verbose=0, penalty='l1')
        parameters = {'clf__C': np.logspace(-5, 1, num=5).tolist(),
                      'ft_red__n_components': np.logspace(3, 1, num=10).astype(int).tolist()}
        pipe = Pipeline([('ft', ft_extractor), ('ft_red', ft_reductor), ('clf', classifier)])
        filename = DEFAULT_TRAIN_LOCATION + "/cv_lr_all_svd_" + utils.timestamp() + ".pkl"
    elif clf == "lr_mixed_svd":
        ft_extractor = preprocessing.create_ft_ctsvd_pd_au()
        classifier = LogisticRegression(verbose=0, penalty='l1')
        parameters = {'clf__C': np.logspace(-2, 2, num=5).tolist(),
                      'ft__ft_extractor__content__reductor__n_components': np.logspace(3.7, 1, num=5).astype(int).tolist()}
        pipe = Pipeline([('ft', ft_extractor), ('clf', classifier)])
        filename = DEFAULT_TRAIN_LOCATION + "/cv_lr_mixed_svd_" + utils.timestamp() + ".pkl"
    elif clf == "rf_all":
        ft_extractor = preprocessing.create_ft_ctsvd_pd_au()
        classifier = RandomForestClassifier()
        parameters = {'clf__max_depth': np.logspace(0, 4, num=5).tolist(),
                      'ft__ft_extractor__content__reductor__n_components':
                          np.logspace(2.5, 1, num=3).astype(int).tolist()}
        pipe = Pipeline([('ft', ft_extractor), ('clf', classifier)])
        filename = DEFAULT_TRAIN_LOCATION + "/cv_lr_rf_all_" + utils.timestamp() + ".pkl"
    else:
        ft_extractor = preprocessing.create_ft_ct()
        classifier = LogisticRegression(verbose=0, penalty='l1')
        parameters = {'clf__C': np.logspace(-2, 2, num=15).tolist()}
        filename = DEFAULT_TRAIN_LOCATION + "/cv_logistic_regression_" + utils.timestamp() + ".pkl"
        pipe = Pipeline([('ft', ft_extractor), ('clf', classifier)])

    # Create the cross-validation search method
    # print pipe.get_params().keys()
    loss = make_scorer(preprocessing.loss_fct, greater_is_better=False)
    gs = GridSearchCV(pipe, parameters, n_jobs=N_PROCESS, verbose=2, scoring=loss)

    # Run the cross-validation
    print "\nPerforming grid search..."
    print "    Pipeline:", [name for name, _ in pipe.steps]
    print "    Parameters: ",
    pprint(parameters)
    print ""
    gs.fit(dataset, target)

    # Save the results
    r = gs.cv_results_
    if clf == "lr" or clf == "lr_all":
        results = (r['param_clf__C'], -r['mean_train_score'], r['std_train_score'], -r['mean_test_score'],
                   r['std_test_score'], r['mean_fit_time'], r['std_fit_time'], clf)
    elif clf == "lr_all_svd":
        results = (r['param_clf__C'], r['param_ft_red__n_components'], -r['mean_train_score'], r['std_train_score'],
                   -r['mean_test_score'], r['std_test_score'], r['mean_fit_time'], r['std_fit_time'], clf)
    elif clf == "lr_mixed_svd":
        results = (r['param_clf__C'], r['param_ft__ft_extractor__content__reductor__n_components'],
                   -r['mean_train_score'], r['std_train_score'], -r['mean_test_score'],
                   r['std_test_score'], r['mean_fit_time'], r['std_fit_time'], clf)
    elif clf == "rf" or clf == "rf_all":
        results = (r['param_clf__max_depth'], -r['mean_train_score'], r['std_train_score'], -r['mean_test_score'],
                   r['std_test_score'], r['mean_fit_time'], r['std_fit_time'], clf)
    else:
        results = (r['param_clf__C'], -r['mean_train_score'], r['std_train_score'], -r['mean_test_score'],
                   r['std_test_score'], r['mean_fit_time'], r['std_fit_time'], clf)
    utils.dump_pickle(results, filename)

    # Print the best individual
    print "\nBest score: %0.3f" % -gs.best_score_
    print "    Best parameters set:"
    best_parameters = gs.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))


def test():

    # 1. Import the features and target
    print "1. Import the features and target\n"
    feature, target = preprocessing.load_ft(preprocessing.DEFAULT_FT_LOCATION + "/ft_l_std_scaler.pkl")
    training_ft, validate_ft, training_target, validate_target = \
        train_test_split(feature, target, test_size=preprocessing.VALIDATE_PART, random_state=preprocessing.SEED)
    print "Training features size: " + str(training_ft.shape) + \
          " and validation features size: " + str(validate_ft.shape)
    print "Training target size: " + str(len(training_target)) + \
          " and validation target size: " + str(len(validate_target)) + "\n"

    # 2. Create the NN
    print "2. Create the Linear Classifier Network"
    penalty = "l1"
    c = 1e-3
    clf = LogisticRegression(penalty=penalty, C=c, verbose=1)
    classes = np.unique(training_target)
    print "\tPenalty: " + str(penalty)
    print "\tC: " + str(c)
    print "\tClasses: " + str(classes) + "\n"

    # 3. Train the Classifier
    print "3. Train the Classifier"
    t_in = time.time()
    clf.fit_transform(training_ft, training_target)
    t_training = time.time() - t_in
    pred_training = clf.predict(training_ft)
    t_in = time.time()
    pred_validate = clf.predict(validate_ft)
    t_validate = time.time() - t_in

    # Compute scores
    score_training = preprocessing.loss_fct(training_ft, pred_training)
    score_validate = preprocessing.loss_fct(validate_ft, pred_validate)

    print "Score on training set: %0.3f and validation set: %0.3f" % (score_training, score_validate)
    print "Time dedicated for training; %0.3fs and for validation: %0.3f" % (t_training, t_validate)


def plot(filename):

    results = utils.load_pickle(filename)
    method = results[-1]
    if method == "lr_all_svd" or method == "lr_mixed_svd" or method == "rf_all":
        c = np.unique(np.array(results[0]))
        k = np.unique(np.array(results[1]))
        st = np.array(results[2]).reshape(len(c), len(k))
        st_err = np.array(results[3]).reshape(len(c), len(k))
        sv = np.array(results[4]).reshape(len(c), len(k))
        sv_err = np.array(results[5]).reshape(len(c), len(k))

        print st, results[2]
        print results[0], c
        print st[0]

        fig, ax1 = plt.subplots()

        for ind, i in enumerate(k):
            ax1.errorbar(c, sv[ind], sv_err[ind], label='k: ' + str(i))
        ax1.set_ylabel('Score')
        ax1.tick_params('y', color=utils.get_style_colors()[0])
        plt.xscale('log')
        fig.tight_layout()
        filename = DEFAULT_TRAIN_LOCATION + "/cv_" + method + ".png"
        ax1.legend(loc=0)
        plt.savefig(filename, format='png', dpi=300)
        plt.close()

    else:
        x = np.array(results[0])
        st = results[1]
        st_err = results[2]
        sv = results[3]
        sv_err = results[4]
        tt = results[5]
        tt_err = results[6]

        fig, ax1 = plt.subplots()
        ax1.errorbar(x, st, st_err, color=utils.get_style_colors()[0])
        ax1.errorbar(x, sv, sv_err, color=utils.get_style_colors()[1])
        if method == "lr":
            ax1.set_xlabel('Evolution of regularization parameter C')
        ax1.set_ylabel('Score')
        ax1.tick_params('y', color=utils.get_style_colors()[0])
        ax2 = ax1.twinx()
        ax2.errorbar(x, tt, tt_err, color=utils.get_style_colors()[2], linewidth=1.5)
        ax2.set_ylabel('Training time (s)', color=utils.get_style_colors()[2])
        ax2.tick_params('y', colors=utils.get_style_colors()[2])
        ax2.grid(b=False)
        plt.xscale('log')
        fig.tight_layout()
        filename = DEFAULT_TRAIN_LOCATION + "/cv_" + method + ".png"
        plt.savefig(filename, format='png', dpi=300)
        plt.close()


if __name__ == '__main__':

    args = sys.argv

    if args[1] == "test":
        test()
    elif args[1] == "gs":
        if len(args) > 2:
            grid_search(args[2])
        else:
            grid_search()
    elif args[1] == "plot":
        plot(args[2])
    else:
        print "Option does not exist. Please, check the preprocessing.py file"
