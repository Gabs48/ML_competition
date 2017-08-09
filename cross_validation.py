
import data
import features
import train
import utils

from copy import copy
from multiprocessing import Pool, Lock, Queue
import numpy as np
import os
import random
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import signal
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

DEFAULT_CV_LOCATION = 'CrossValidation'
TRAINING_PART = 0.95
VALIDATE_PART = 1 - TRAINING_PART
RANDOM = 42
FRACTION = 0.5
N_PROCESS = 4

# Global variables
writing_mutex = Lock()
dataset = data.load_pickled_data()


class KeyboardInterruptError(Exception): pass


def experience(params):

	try:
		# Set cross-validation parameters
		n = params[0]
		min_df = params[1]
		max_df = params[2]
		seed = params[3]
		filename = params[4]

		# Set the model
		training_set, validate_set = train_test_split(dataset["train"],
			test_size=VALIDATE_PART, random_state=seed)
		training_set = training_set
		target_training = train.create_target(training_set)
		target_validate = train.create_target(validate_set)

		ft_extractor = features.ReviewsFeaturesExtractor(ngram=n, min_df=min_df, max_df=max_df)
		classifier = LogisticRegression(verbose=0)
		pipe = Pipeline([('ft_extractor', ft_extractor), ('classifier', classifier)])

		# Train and validate the model
		t_in = time.time()
		pipe.fit_transform(training_set, target_training)
		t_training = time.time() - t_in
		pred_training = pipe.predict(training_set)
		t_in = time.time()
		pred_validate = pipe.predict(validate_set)
		t_validate = time.time() - t_in

		# Compute scores
		score_training = train.loss_fct(target_training, pred_training)
		score_validate = train.loss_fct(target_validate, pred_validate)

		cv_res = (n, min_df, max_df, seed, score_training, score_validate, t_training, t_validate)

		with writing_mutex:
			if os.path.exists(filename):
				cv_old_res = utils.load_pickle(filename)
				cv_old_res.append(cv_res)
				utils.dump_pickle(cv_old_res, filename)
			else:
				utils.dump_pickle([cv_res], filename)

	except KeyboardInterrupt:
		raise KeyboardInterruptError()

	return cv_res


def perform_cross_validation(filename=None):

	cv_res = []
	if filename is None:
		filename = os.path.join(DEFAULT_CV_LOCATION, "cross_validation_" + utils.timestamp() + ".pkl")

	# Create a pool of processes
	pool = Pool(N_PROCESS)

	# Create the parameters for a cross-validation experiment
	parameters = []
	for n in [1, 2, 3, 4]:
		for min_df in np.logspace(-5, -1, num=10):
			for max_df in [0.3]:
				for seed in [random.randint(0, 1000) for i in xrange(6)]:
					parameters.append((n, min_df, max_df, seed, filename))

	# Execute pool of processes
	try:
		cv_res = pool.map(experience, parameters)
		utils.dump_pickle(cv_res, filename)
		pool.close()
	except KeyboardInterrupt:
		print("Caught KeyboardInterrupt, terminating workers")
		pool.terminate()
	except Exception, e:
		print("Caught exception %r, terminating workers" % (e,))
		pool.terminate()
	finally:
		pool.join()

	print cv_res, len(cv_res)


def plot_cross_validation(filename):

	results = utils.load_pickle(filename)
	print len(results)

	# Transcript results
	score_training_ma = dict()
	score_validate_ma = dict()
	t_training_ma = dict()
	t_validate_ma = dict()
	score_training_mi = dict()
	score_validate_mi = dict()
	t_training_mi = dict()
	t_validate_mi = dict()

	for r in results:
		if r[0] not in score_training_ma:
			score_training_ma[r[0]] = dict()
			score_validate_ma[r[0]] = dict()
			t_training_ma[r[0]] = dict()
			t_validate_ma[r[0]] = dict()
			score_training_mi[r[0]] = dict()
			score_validate_mi[r[0]] = dict()
			t_training_mi[r[0]] = dict()
			t_validate_mi[r[0]] = dict()
		else:
			if r[1] not in score_training_ma[r[0]]:
				score_training_ma[r[0]][r[1]] = dict()
				score_validate_ma[r[0]][r[1]] = dict()
				t_training_ma[r[0]][r[1]] = dict()
				t_validate_ma[r[0]][r[1]] = dict()
			else:
				if r[2] not in score_training_ma[r[0]][r[1]]:
					score_training_ma[r[0]][r[1]][r[2]] = []
					score_validate_ma[r[0]][r[1]][r[2]] = []
					t_training_ma[r[0]][r[1]][r[2]] = []
					t_validate_ma[r[0]][r[1]][r[2]] = []
				else:
					score_training_ma[r[0]][r[1]][r[2]].append(r[4])
					score_validate_ma[r[0]][r[1]][r[2]].append(r[5])
					t_training_ma[r[0]][r[1]][r[2]].append(r[6])
					t_validate_ma[r[0]][r[1]][r[2]].append(r[7])

			if r[2] not in score_training_mi[r[0]]:
				score_training_mi[r[0]][r[2]] = dict()
				score_validate_mi[r[0]][r[2]] = dict()
				t_training_mi[r[0]][r[2]] = dict()
				t_validate_mi[r[0]][r[2]] = dict()
			else:
				if r[1] not in score_training_mi[r[0]][r[2]]:
					score_training_mi[r[0]][r[2]][r[1]] = []
					score_validate_mi[r[0]][r[2]][r[1]] = []
					t_training_mi[r[0]][r[2]][r[1]] = []
					t_validate_mi[r[0]][r[2]][r[1]] = []
				else:
					score_training_mi[r[0]][r[2]][r[1]].append(r[4])
					score_validate_mi[r[0]][r[2]][r[1]].append(r[5])
					t_training_mi[r[0]][r[2]][r[1]].append(r[6])
					t_validate_mi[r[0]][r[2]][r[1]].append(r[7])

	# Plot max_df evolution
	for n in score_training_ma:
		for min_df in score_training_ma[n]:
			st_d = score_training_ma[n][min_df]
			sv_d = score_validate_ma[n][min_df]
			st_k = sorted(st_d.iterkeys())
			x = [max_df for max_df in st_k]
			st = [np.mean(st_d[max_df]) for max_df in st_k]
			st_err = [np.std(st_d[max_df]) for max_df in st_k]
			sv = [np.mean(sv_d[max_df]) for max_df in st_k]
			sv_err = [np.std(sv_d[max_df]) for max_df in st_k]
			plt.errorbar(x, st, st_err)
			plt.errorbar(x, sv, sv_err)
			plt.tight_layout()
			plt.savefig(DEFAULT_CV_LOCATION + "/cv_maxdf_" + str(n) + "_" + str(min_df) + ".png", format='png', dpi=300)
			plt.close()

	# Plot max_df evolution
	for n in score_training_mi:
		for max_df in score_training_mi[n]:
			print n, max_df, score_training_mi[n][max_df]
			st_d = score_training_mi[n][max_df]
			sv_d = score_validate_mi[n][max_df]
			st_k = sorted(st_d.iterkeys())
			x = [min_df for min_df in st_k]
			st = [np.mean(st_d[min_df]) for min_df in st_k]
			st_err = [np.std(st_d[min_df]) for min_df in st_k]
			sv = [np.mean(sv_d[min_df]) for min_df in st_k]
			sv_err = [np.std(sv_d[min_df]) for min_df in st_k]
			plt.errorbar(x, st, st_err)
			plt.errorbar(x, sv, sv_err)
			plt.tight_layout()
			plt.savefig(DEFAULT_CV_LOCATION + "/cv_mindf_" + str(n) + "_" + str(max_df) + ".png", format='png', dpi=300)
			plt.close()


if __name__ == '__main__':

	args = sys.argv

	if args[1] == "compute":
		perform_cross_validation()
	elif args[1] == "plot":
		plot_cross_validation(args[2])
	else:
		print "Please, possible uses are 'python cross_validation.py compute' or " \
			"'python cross_validation.py plot file.pkl'"