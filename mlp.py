
import features
import data
import train
import utils

import numpy as np
from pprint import pprint
import sys

from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

TRAINING_PART = 0.95
VALIDATE_PART = 1 - TRAINING_PART
SEED = 189
BATCH_SIZE = 200
N_PROCESS = 10
DEFAULT_TRAIN_LOCATION = 'Train'


def create_ft():

    # 1. Get the data
    print "1. Load data\n"
    dataset = data.load_pickled_data()['train']
    target = train.create_target(dataset)

    # 2. Create the feature matrices
    print "2. Create features"
    ft_pipe = Pipeline([('ft', features.create_ft_ct_pd_au()), ('norm', StandardScaler(with_mean=False))])
    feature = ft_pipe.fit_transform(dataset)
    print "Features matrix size: " + str(feature.shape) + " and target size: " + str(len(target)) + "\n"

    # 3. Save features
    print "3. Save features"
    features.save_ft(feature, target, filename=features.DEFAULT_FT_LOCATION + "/mlp_ft.pkl")


def test():

    # 1. Import the features and target
    print "1. Import the features and target\n"
    feature, target = features.load_ft(features.DEFAULT_FT_LOCATION + "/mlp_ft.pkl")
    training_ft, validate_ft, training_target, validate_target = \
        train_test_split(feature, target, test_size=VALIDATE_PART, random_state=SEED)
    print "Training features size: " + str(training_ft.shape) +\
          " and validation features size: " + str(validate_ft.shape)
    print "Training target size: " + str(len(training_target)) + \
          " and validation target size: " + str(len(validate_target)) + "\n"

    # 2. Create the NN
    print "2. Create the Neural Network"
    batch_size = BATCH_SIZE
    clf = MLPClassifier(alpha=1e-3, hidden_layer_sizes=(50,), batch_size=batch_size, warm_start=True)
    classes = np.unique(training_target)
    print "Classes: " + str(classes) + "\n"

    # 3. Train the NN
    print "3. Train the Neural Network"
    for i in xrange(0, training_ft.shape[0]/batch_size):
        batch_ft = training_ft[i*batch_size:(i+1)*batch_size]
        batch_target = training_target[i*batch_size:(i+1)*batch_size]

        if i == 0:
            clf = clf.partial_fit(batch_ft, batch_target, classes)
        else:
            clf = clf.partial_fit(batch_ft, batch_target)

        print "Iteration %i Score on training set: %0.3f and validation set: %0.3f" % \
              (i, train.loss_fct(clf.predict(training_ft), training_target),
               train.loss_fct(clf.predict(validate_ft), validate_target))


def grid_search():

    # 1. Import the features and target
    print "1. Import the features and target\n"
    feature, target = features.load_ft(features.DEFAULT_FT_LOCATION + "/mlp_ft.pkl")
    training_ft, validate_ft, training_target, validate_target = \
        train_test_split(feature, target, test_size=VALIDATE_PART, random_state=SEED)
    print "Training features size: " + str(training_ft.shape) + \
          " and validation features size: " + str(validate_ft.shape)
    print "Training target size: " + str(len(training_target)) + \
          " and validation target size: " + str(len(validate_target)) + "\n"

    # 2. Create the Neural Network
    print "2. Create the Neural Network\n"
    batch_size = BATCH_SIZE
    clf = MLPClassifier(alpha=1e-3, hidden_layer_sizes=(50,), batch_size=batch_size, warm_start=True)
    parameters = {'alpha': np.logspace(-5, -1, num=4).tolist(),
                  'hidden_layer_sizes': [(i,) for i in [10, 20, 50, 100]],
                  'batch_size': [50, 100, 500, 1000]}

    # 3. Create and run the cross-validation search method
    print "3. Perform grid search"
    print "    Parameters: ",
    pprint(parameters)
    print ""

    loss = make_scorer(train.loss_fct, greater_is_better=False)
    gs = GridSearchCV(clf, parameters, n_jobs=N_PROCESS, verbose=2, scoring=loss)
    gs.fit(training_ft, training_target)

    # 4. Save the results
    print "4. Save and plot scores"
    r = gs.cv_results_
    utils.dump_pickle(r, DEFAULT_TRAIN_LOCATION + "/cv_mlp_" + utils.timestamp() + ".pkl")

    print "\nBest score: %0.3f" % -gs.best_score_
    print "Best parameters set:"
    best_parameters = gs.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))


if __name__ == '__main__':

    args = sys.argv

    if args[1] == "test":
        test()
    elif args[1] == "ft":
        create_ft()
    elif args[1] == "gs":
        grid_search()
    else:
        print "Option does not exist. Please, check the feature_selection.py file"
