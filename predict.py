"""Sample script creating some baseline predictions."""

import data
import preprocessing
import linear
import nonlinear
import utils

import sys


def main(ft_filename=None):

	print "1. Load the data"
	dataset = data.load_pickled_data()
	train_set = dataset['train']
	train_target = linear.create_target(train_set)
	test_set = dataset['test']
	print "Train and Test sets lengths: " + str(len(train_set)) + " " + str(len(test_set)) + "\n"

	print "2. Compute the features\n"
	if ft_filename is None:
		preprocessing.create_ft()

	print "3. Train the model"
	nonlinear.pre

	print "4. Save predictions'
	predictions = pipe.predict(test_set)
	save_prd(predictions)


if __name__ == '__main__':

	args = sys.argv

	if args > 1:
		main(args[1])
	else:
		main()
