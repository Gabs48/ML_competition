"""Module containing utility functions."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import os
import time


try:
	import cPickle as pickle
except:
	print "Warning: Couldn't import cPickle, using native pickle instead."
	import pickle


## Disk access function ##

def _make_dir(path):
	path_dir = os.path.dirname(path)
	if not os.path.exists(path_dir):
		os.makedirs(path_dir)


def make_dir(path):
	if not os.path.exists(path):
		os.makedirs(path)		


def dump_pickle(data, path):
	"""Dumps data to a pkl file."""
	if not path.endswith('.pkl'):
		raise ValueError(
				'Pickle files should end with .pkl, but got %s instead' % path)
	_make_dir(path)
	with open(path, 'wb') as pkl_file:
		pickle.dump(data, pkl_file, pickle.HIGHEST_PROTOCOL)


def load_pickle(path_to_pickle):
	with open(path_to_pickle, 'rb') as pkl_file:
		return pickle.load(pkl_file)


def dump_npy(array, path):
	"""Dumps a single numpy array to a npy file."""
	if not path.endswith('.npy'):
		raise ValueError(
				'Filename should end with .npy, but got %s instead' % path)
	_make_dir(path)
	with open(path, 'wb') as npy_file:
		np.save(npy_file, array)


def load_npy(path):
	return np.load(path)


## Formatting functions ##

def timestamp():
		return time.strftime("%Y%m%d-%H%M%S", time.localtime())


def generate_unqiue_file_name(basename, file_ext):
	return basename + '_' + timestamp() + '.' + file_ext


def n2str(num):
	""" convert a number into a short string"""
	if abs(num) < 1 and abs(num) > 1e-50 or abs(num) > 1E4:
		numFormat =  ".2e"
	elif abs(round(num) - num) < 0.001 or abs(num) > 1E4:
		numFormat = ".0f"
	elif abs(num) > 1E1:
		numFormat = ".1f"
	else:
		numFormat = ".2f"
	return ("{:" + numFormat + "}").format(num)


def to_percent(y, position):
	"""
	This formatter transforms a matplotlib axis from values beween 0 and 1
	to percents.
	"""
	s = str(100 * y)

	# The percent symbol needs escaping in latex
	if matplotlib.rcParams['text.usetex'] is True:
		return s + r'$\%$'
	else:
		return s + '%'


## Mathematical operation functions ##


def lp_filter(array, window):

	averageArr = np.convolve(array, np.ones((window,))/window, mode='valid')

	return averageArr

## Matplotlib functions ##

plt.style.use('fivethirtyeight')


def get_style_colors():
	""" Return a arry with the current style colors """

	if 'axes.prop_cycle' in plt.rcParams:
		cols = plt.rcParams['axes.prop_cycle']
		col_list = []
		for v in cols:
			col_list.append(v["color"])
	else:
		col_list = ['b', 'r', 'y', 'g', 'k']
	return col_list