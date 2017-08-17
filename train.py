""" Script that transform the content of train in sklearn boolean features """

from features import *
import data
import utils

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
from pprint import pprint

from sklearn.decomposition import TruncatedSVD, SparsePCA, PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import make_scorer
from sklearn.model_selection import *
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier
import sys

# Configuration
plt.style.use('fivethirtyeight')

# Global constants
DEFAULT_MODEL_LOCATION = 'Models'
DEFAULT_FT_LOCATION = 'Features'
DEFAULT_TRAIN_LOCATION = 'Train'
DEFAULT_FT_PATH = os.path.join(DEFAULT_FT_LOCATION, 'features.pkl')
DEFAULT_FT_MODEL_PATH = os.path.join(DEFAULT_FT_LOCATION, 'features_model.pkl')
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODEL_LOCATION, 'model.pkl')
TRAINING_PART = 0.95
VALIDATE_PART = 1 - TRAINING_PART
RANDOM = 42
N_PROCESS = 4


def loss_fct(truth, prediction):
	"""
	Evaluate the gap between the target and prediction
	"""

	diff = truth - prediction
	score = float(np.sum(np.abs(diff))) / diff.size

	return score


def create_target(dataset):

	target = []
	for r in dataset:
		target.append(r.rating)

	return np.array(target)


def optimize_classifier(clf='logistic_regression'):

	# Get the data
	print "Load data"
	dataset = data.load_pickled_data()["train"]
	target = create_target(dataset)

	# Create the pipe
	ft_extractor = create_ft_ct_pd_au()
	if clf == "logistic_regression":
		classifier = LogisticRegression(verbose=0, penalty='l2')
		parameters = {'clf__C': np.logspace(-2, 2, num=15).tolist()}
		filename = DEFAULT_TRAIN_LOCATION + "/cv_logistic_regression_" + utils.timestamp() + ".pkl"
	else:
		classifier = LogisticRegression(verbose=0, penalty='l2')
		parameters = {'clf__C': np.logspace(-2, 2, num=15).tolist()}
		filename = DEFAULT_TRAIN_LOCATION + "/cv_logistic_regression_" + utils.timestamp() + ".pkl"

	pipe = Pipeline([('ft', ft_extractor), ('clf', classifier)])

	# Create the cross-validation search method
	loss = make_scorer(loss_fct, greater_is_better=False)
	grid_search = GridSearchCV(pipe, parameters, n_jobs=N_PROCESS, verbose=2, scoring=loss)

	# Run the cross-validation
	print "\nPerforming grid search..."
	print "    Pipeline:", [name for name, _ in pipe.steps]
	print "    Parameters: ",
	pprint(parameters)
	grid_search.fit(dataset, target)

	# Save the results
	r = grid_search.cv_results_
	results = (r['param_clf__C'], -r['mean_train_score'], r['std_train_score'], -r['mean_test_score'],
			r['std_test_score'], r['mean_fit_time'], r['std_fit_time'], clf)
	utils.dump_pickle(results, filename)

	# Print the best individual
	print "\nBest score: %0.3f" % -grid_search.best_score_
	print "    Best parameters set:"
	best_parameters = grid_search.best_estimator_.get_params()
	for param_name in sorted(parameters.keys()):
		print("\t%s: %r" % (param_name, best_parameters[param_name]))


def plot_optimization(filename):

	results = utils.load_pickle(filename)

	x = np.array(results[0])
	st = results[1]
	st_err = results[2]
	sv = results[3]
	sv_err = results[4]
	tt = results[5]
	tt_err = results[6]
	method = results[7]

	fig, ax1 = plt.subplots()
	ax1.errorbar(x, st, st_err, color=utils.get_style_colors()[0])
	ax1.errorbar(x, sv, sv_err, color=utils.get_style_colors()[1])
	if method == "logistic_regression":
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


def test_linear_classifiers():

	print "Load data and compute features"
	dataset = data.load_pickled_data()
	training_set, validate_set = train_test_split(dataset["train"],
		test_size=VALIDATE_PART, random_state=356)

	training_target = create_target(training_set)
	validate_target = create_target(validate_set)

	ft_extractor = utils.load_pickle(DEFAULT_FT_MODEL_PATH)
	training_ft = ft_extractor.transform(training_set)
	validate_ft = ft_extractor.transform(validate_set)

	# Declare classifiers
	names = ["Linear Discriminant Analysis", "Logistic Regression", "Multinomial Naive Bayes",
			"Linear Support Vector Classifier", "Perceptron", "Ridge Classifier"]
	classifiers = [
		LinearDiscriminantAnalysis(),
		LogisticRegression(),
		MultinomialNB(),
		LinearSVC(),
		Perceptron(),
		RidgeClassifier()]
	training_scores = []
	validate_scores = []

	# Loop over classifiers
	for name, clf in zip(names, classifiers):

		# Train classifier
		print "Training classifier " + name
		pipe = Pipeline([("pca", TruncatedSVD(5000)), ("classifier", clf)])
		pipe.fit(training_ft, training_target)
		training_predict = pipe.predict(training_ft)
		validate_predict = pipe.predict(validate_ft)
		training_scores.append(loss_fct(training_target, training_predict))
		validate_scores.append(loss_fct(validate_target, validate_predict))
		print "Scores: " + str(training_scores[-1]) + " " + str(validate_scores[-1])


if __name__ == '__main__':

	args = sys.argv

	if args[1] == "test":
		test_linear_classifiers()
	elif args[1] == "opt":
		optimize_classifier()
	elif args[1] == "plot":
		plot_optimization(args[2])
	else:
		print "Option does not exist. Please, check the features.py file"

