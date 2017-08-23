"""Sample script creating some baseline predictions."""

import create_submission
import data
import preprocessing
import utils
import linear

import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import sys


LR_PREDICTIONS_BASENAME = os.path.join('Predictions', 'logistic_regression')


def save_prd(predictions, path=LR_PREDICTIONS_BASENAME):

	filename_npy = utils.generate_unqiue_file_name(path, 'npy')
	filename_csv = utils.generate_unqiue_file_name(path, 'csv')

	utils.dump_npy(predictions, filename_npy)
	create_submission.write_predictions_to_csv(predictions, filename_csv)


def main():

	print 'Load the data'
	dataset = data.load_pickled_data()
	train_set = dataset['train']
	test_set = dataset['test']
	print "Train and Test sets lengths: " + str(len(train_set)) + " " + str(len(test_set))
	train_target = linear.create_target(train_set)

	# Model
	ft_extractor = preprocessing.create_ft_ct_pd_au(ngram=3, max_df=0.3, min_df=0.0001, w_ct=1, w_pd=1, w_au=1)
	classifier = LogisticRegression()
	pipe = Pipeline([('ft_extractor', ft_extractor), ('classifier', classifier)])

	# Train the model
	print "Train the model"
	pipe.fit_transform(train_set, train_target)

	# Predict the test set and save
	print 'Save predictions'
	predictions = pipe.predict(test_set)
	save_prd(predictions)


if __name__ == '__main__':

	args = sys.argv
	main()
