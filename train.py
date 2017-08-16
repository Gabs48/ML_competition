""" Script that transform the content of train in sklearn boolean features """

from features import *
import data
import utils

import numpy as np
import os
from sklearn.model_selection import *
from sklearn.decomposition import TruncatedSVD, SparsePCA, PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier
import sys

DEFAULT_MODEL_LOCATION = 'Models'
DEFAULT_FT_LOCATION = 'Features'
DEFAULT_FT_PATH = os.path.join(DEFAULT_FT_LOCATION, 'features.pkl')
DEFAULT_FT_MODEL_PATH = os.path.join(DEFAULT_FT_LOCATION, 'features_model.pkl')
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODEL_LOCATION, 'model.pkl')
TRAINING_PART = 0.95
VALIDATE_PART = 1 - TRAINING_PART
RANDOM = 42


def loss_fct(target, prediction):
	"""
	Evaluate the gap between the target and prediction
	"""

	diff = target - prediction
	score = float(np.sum(np.abs(diff))) / diff.size

	return score


def create_target(dataset):

	target = []
	for r in dataset:
		target.append(r.rating)

	return np.array(target)


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


def create_classifier():

	dataset = data.load_pickled_data()

	# Get features and target
	target = create_target(dataset)
	features = utils.load_pickle(DEFAULT_FT_PATH)

	# Perform linear regression
	model, score = create_predict_model(ft, target)


if __name__ == '__main__':

	args = sys.argv

	if args[1] == "test":
		test_linear_classifiers()
	elif args[1] == "compute":
		create_classifier()
	else:
		print "Option does not exist. Please, check the features.py file"

