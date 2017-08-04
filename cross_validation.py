
import data
import features
import train
import utils

import os
import random
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

DEFAULT_CV_LOCATION = 'CrossValidation'
TRAINING_PART = 0.95
VALIDATE_PART = 1 - TRAINING_PART
RANDOM = 42
FRACTION = 0.5


def perform_cross_validation(filename=None):

	cross_validation = []
	if filename is None:
		filename = os.path.join(DEFAULT_CV_LOCATION, "cross_validation.pkl")

	# Iterate on the ngrams length
	for n in range(1,4):

		# Iterate on the lowest boundary in the ngram-dataset
		for lb in [0.1, 0.2, 0.3, 0.4, 0.5]:

			# Iterate on the training/validation split
			for seed in [random.randint(0, 1000) for i in xrange(5)]:

				# Get the data
				dataset = data.load_pickled_data()

				# Set the model
				training_set, validate_set = train_test_split(dataset["train"], test_size=VALIDATE_PART, random_state=seed)
				target_training = train.create_target(training_set)
				target_validate = train.create_target(validate_set)

				ft_extractor = features.ReviewsFeaturesExtractor(ngram=n, frac=FRACTION, low_boundary=lb)
				classifier = LogisticRegression(verbose=0)
				pipe = Pipeline([('ft_extractor', ft_extractor), ('classifier', classifier)])

				# Train and validate the model
				pipe.fit_transform(training_set, target_training)
				pred_training = pipe.predict(training_set)
				pred_validate = pipe.predict(validate_set)

				score_training = train.loss_fct(target_training, pred_training)
				score_validate = train.loss_fct(target_validate, pred_validate)

				cross_validation.append({"n": n, "lb": lb, "seed": seed, "st":score_training, "sv": score_validate})
				utils.dump_pickle(cross_validation, filename)


if __name__ == '__main__':

	perform_cross_validation()