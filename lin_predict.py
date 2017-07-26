"""Sample script creating some baseline predictions."""

import data
import features
import utils
import validate

import os
import numpy as np
import sys



def predict_lr(dataset, ft_mdl_path, lr_mdl_path):
	"""
	Predict scores on the dataset test using model in lr_mdl_path and
	training feature model in ft_mdl_path
	"""

	print "\n -- PREDICT TEST REVIEWS --"

	print " -- Compute test features -- "
	ft_mdl = validate.get_model(path=ft_mdl_path)
	test_ft_dict = validate.get_test_ft(dataset, ft_mdl)
	test_ft = ft_mdl.transform(test_ft_dict)
		
	print " -- Get model and estimate ratings -- "
	lr_mdl = validate.get_model(path=lr_mdl_path)
	predictions = lr_mdl.predict(test_ft)

	return predictions