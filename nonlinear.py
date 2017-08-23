
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

from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline

BATCH_SIZE = 1000
N_PROCESS = 10


def test(predict=False):

    # 1. Import the features and target
    print "1. Import the features and target\n"
    test_ft, train_ft, target = preprocessing.load_ft(preprocessing.DEFAULT_FT_LOCATION + "/ft_max_scaler.pkl")
    recenter = False
    dilatation = 1
    train_ft = train_ft * dilatation
    test_ft = test_ft * dilatation
    if recenter:
        norm_factor = train_ft.data.mean()
        train_ft.data -= norm_factor
        test_ft.data -= norm_factor
    training_ft, validate_ft, training_target, validate_target = \
        train_test_split(train_ft, target, test_size=preprocessing.VALIDATE_PART, random_state=preprocessing.SEED)
    print "Training features size: " + str(training_ft.shape) +\
          " and validation features size: " + str(validate_ft.shape)
    print "Training target size: " + str(len(training_target)) + \
          " and validation target size: " + str(len(validate_target)) + "\n"

    # 2. Create the NN
    print "2. Create the Neural Network"
    batch_size = BATCH_SIZE
    layers = (15,)
    activation = "tanh"
    alpha = 1e-7
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
    s += "\n\tRe-centering: " + str(recenter)
    s += "\n\tData dilatation factor: " + str(dilatation)
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
        ax1.plot(training_score, color=utils.get_style_colors()[0])
        ax1.plot(validate_score, color=utils.get_style_colors()[1])
        ax1.set_ylabel('Score')
        fig.tight_layout()
        filename = linear.DEFAULT_TRAIN_LOCATION + "/MLP/cv_mlp_" + str(utils.timestamp())
        utils.dump_pickle((training_score, validate_score), filename + ".pkl")
        text_file = open(filename + ".txt", "w")
        text_file.write(s)
        text_file.close()
        plt.savefig(filename + ".png", format='png', dpi=300)
        plt.close()

    if predict:
        # 3. Train on full dataset
        print "3. Train\n"
        clf = clf.fit(train_ft, target)

        # 4. Predict test results
        print "4. Predict test results"
        test_pred = lab.transform(clf.predict(test_ft))
        train_pred = lab.transform(clf.predict(train_ft))
        print "Score on training set: %0.3f" % preprocessing.loss_fct(train_pred, training_target)
        
        filename = linear.DEFAULT_PRED_LOCATION + "/mlp_" + str(utils.timestamp())
        utils.dump_pickle(test_pred, filename + ".pkl")
        create_submission.write_predictions_to_csv(test_pred, filename + ".csv")


def grid_search():

    # 1. Import the features and target
    print "1. Import the features and target\n"
    feature, target = preprocessing.load_ft(preprocessing.DEFAULT_FT_LOCATION + "/ft.pkl")
    training_ft, validate_ft, training_target, validate_target = \
        train_test_split(feature, target, test_size=preprocessing.VALIDATE_PART, random_state=preprocessing.SEED)
    print "Training features size: " + str(training_ft.shape) + \
          " and validation features size: " + str(validate_ft.shape)
    print "Training target size: " + str(len(training_target)) + \
          " and validation target size: " + str(len(validate_target)) + "\n"

    # 2. Create the Neural Network
    print "2. Create the Neural Network\n"
    batch_size = BATCH_SIZE
    clf = MLPClassifier(alpha=1e-3, hidden_layer_sizes=(50,), batch_size=batch_size, warm_start=True)
    parameters = {'alpha': np.logspace(-5, -1, num=5).tolist(),
                  'hidden_layer_sizes': [(i,) for i in [10, 50, 100, 200, 400]],
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
    st = -r['mean_train_score']
    st_err  = r['std_train_score']
    sv = -r['mean_test_score']
    sv_err = r['std_test_score']
    tt =r['mean_fit_time']
    tt_err = r['std_fit_time']

    alpha_r = np.unique(alpha)
    layers_r = np.unique(layers)
    batch_size_r = np.unique(batch_size)

    sv = sv.reshape(len(batch_size_r), len(layers_r), len(alpha_r))

    for i in xrange(len(batch_size_r)):
        score = sv[i, :, :]
        plt.imshow(score, interpolation='nearest', cmap=plt.cm.hot,
                   norm=utils.MidpointNormalize(vmin=0.5, midpoint=0.6))
        plt.ylabel('First layer units')
        plt.xlabel('Regularization parameter alpha')
        plt.colorbar()
        plt.xticks(np.arange(len(alpha_r)), alpha_r, rotation=45)
        plt.yticks(np.arange(len(layers_r)), layers_r)
        plt.title('Validation score')
        plt.tight_layout()
        filename = linear.DEFAULT_TRAIN_LOCATION + "/cv_mlp_hotmap.png"
        plt.savefig(filename, format='png', dpi=300)
        plt.close()


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
    else:
        print "Option does not exist. Please, check the feature_selection.py file"
