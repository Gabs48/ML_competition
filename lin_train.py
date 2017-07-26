""" Script that transform the content of train in sklearn boolean features """

import data
import features
import utils

import numpy as np
import os
from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import sys


def loss_fct(target, prediction):
	"""
	Evaluate the gap between the target and prediction
	"""

	diff = target - prediction
	score = float(np.sum(np.abs(diff))) / diff.size

	return score


def create_model(ft, target, verbose=0):
	"""
	Extract features in two steps:
	 - Creation of a vect dict
	 - PCA to reduce dimensionality
	"""

	print "\n -- CREATE A LOGISTIC REGRESSION MODEL --"

	# Create a sklearn pipe
	clf = LogisticRegression(verbose=verbose)
	pipe = Pipeline([('classifier', clf)])

	# Apply pipe
	pipe.fit_transform(ft, target)

	# Verify
	predict = pipe.predict(ft)
	score = loss_fct(target, predict)
	print " -- Model created with Mean Absolute Error = " + \
		str(score) + " --"

	return pipe, score