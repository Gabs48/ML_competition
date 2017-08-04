""" Script that transform the content of train in sklearn boolean features """

import data
import utils

import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import *
from sklearn.pipeline import Pipeline
import sys

DEFAULT_MODEL_LOCATION = 'Models'
DEFAULT_FT_LOCATION = 'Features'
DEFAULT_FT_PATH = os.path.join(DEFAULT_FT_LOCATION, 'features.pkl')
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODEL_LOCATION, 'model.pkl')
TRAINING_PART = 0.95
TEST_PART = 1 - TRAINING_PART
RANDOM = 42


def loss_fct(target, prediction):
	"""
	Evaluate the gap between the target and prediction
	"""

	diff = target - prediction
	score = float(np.sum(np.abs(diff))) / diff.size

	return score


def create_predict_model(ft, target, verbose=0):
	"""
	Extract features in two steps:
	 - Creation of a vect dict
	 - PCA to reduce dimensionality
	"""

	print "\n -- CREATE A LOGISTIC REGRESSION MODEL --"

	# Create a sklearn pipe
	clf = LogisticRegression(verbose=verbose)
	pipe = Pipeline([('classifier', clf)])
	print "Feature matrix size: " + str(ft.shape) + " and target size: " + str(target.shape)

	# Apply pipe
	pipe.fit_transform(ft, target)

	# Verify
	predict = pipe.predict(ft)
	score = loss_fct(target, predict)
	print " -- Model created with Mean Absolute Error = " + \
		str(score) + " --"
	print ' Prediction: ' + str(predict)
	print ' Target: ' + str(target)

	return pipe, score


def create_target(dataset):

	target = []
	for review in dataset:
		target.append(review.rating)

	return np.array(target)


def get_ft(mini=None, maxi=None, path=DEFAULT_FT_PATH):
	"""
	Load feature vectors
	"""
	ft = utils.load_pickle(path)
	if maxi == None:
		maxi = ft.shape[0]
	if mini == None:
		mini = 0

	return ft #[-mini:maxi, :]


def _parse_args(args):
	"""
	Parse argument for validation
	"""

	if not len(args) in (1, 2):
		print ('Usage: python2 train.py <path_to_features.pkl> ')
		return

	ft_path = sys.argv[1] if len(sys.argv) == 2 else DEFAULT_FT_PATH

	return ft_path


def main(ft_path):

	dataset = data.load_pickled_data()
	train_set, test_set = train_test_split(dataset["train"], test_size=TEST_PART, random_state=RANDOM)

	# Get features and target
	target = create_target(train_set)
	ft = get_ft(path=ft_path)

	# Perform linear regression
	model, score = create_predict_model(ft, target)


if __name__ == '__main__':

	args = sys.argv
	ft_path = _parse_args(args)
	main(ft_path)