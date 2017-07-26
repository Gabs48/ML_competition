 
"""Sample script creating some baseline predictions."""

import data
import features
import utils
import validate
import lin_predict
import nonlin_predict

import os
import numpy as np
import sys


LR_PREDICTIONS_BASENAME = os.path.join('Predictions', 'logistic_regression')


def save_prd(prd, path=LR_PREDICTIONS_BASENAME):
	"""
	Save predictions in a npy file
	"""

	prd_file_name = utils.generate_unqiue_file_name(
		LR_PREDICTIONS_BASENAME, 'npy')
	utils.dump_npy(prd, prd_file_name)
	print 'Dumped predictions to %s' % prd_file_name


def _parse_args(args):
	"""
	Parse argument for test prediction
	"""

	if not len(args) in (2, 3):
		print ('Usage: python2 predict.py <path_to_ft_model.plk> <path_to_train_model.pkl> ')
		return

	ft_mdl_path = sys.argv[1]
	lr_mdl_path = sys.argv[2] if len(sys.argv) == 3 else validate.DEFAULT_MODEL_PATH
	
	return ft_mdl_path, lr_mdl_path


def main(ft_mdl_path, lr_mdl_path):
	dataset = data.load_pickled_data()
	test_data = dataset['test']

	# Predict
	prd = predict_lr(test_data, ft_mdl_path, lr_mdl_path)

	# Save predictions
	save_prd(prd)
	

if __name__ == '__main__':
	args = sys.argv
	arguments = _parse_args(args)
	if arguments:
		main(*arguments)