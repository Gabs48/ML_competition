"""Script loading the pickle files and creating a set of features to be processed."""

import analyze
import data
import utils


import numpy as np
import os
from scipy.sparse import hstack, csr_matrix, lil_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import text
from sklearn.model_selection import *
import sys

reload(sys)  
sys.setdefaultencoding('utf8')

DEFAULT_FT_LOCATION = 'Features'
TRAINING_PART = 0.95
VALIDATE_PART = 1 - TRAINING_PART
RANDOM = 42


class ReviewsFeaturesExtractor(BaseEstimator, TransformerMixin):
	"""
	Custom SKLearn estimator that extract features from a dataset
	"""

	def __init__(self, frac=0.59, low_boundary=0.4, ngram=4):

		self.frac = frac
		self.low_boundary = low_boundary
		self.ngram = ngram
		self.features_names = np.array([])
		self.dataset = None

	def fit(self, x, y=None):

		return self

	def transform(self, dataset):
		"""
		Extract and combine all features
		"""

		# Extract feature matrices
		print "Process the text"

		ct_ft, ct_nm = self.content_ft(dataset)

		print "Fill the features matrices"

		# Training phase
		if self.features_names.size == 0:
			features_matrix = ct_ft
			self.features_names = ct_nm

		# Test and validation phase
		else:
			features_matrix = lil_matrix((len(dataset), self.features_names.size))
			# print ct_ft.tocsr().shape, features_matrix.shape, np.in1d(self.features_names, ct_nm).shape, np.in1d(ct_nm, self.features_names).shape
			for row in range(features_matrix.shape[0]):
				features_matrix[row, np.in1d(self.features_names, ct_nm)] = ct_ft.tolil()[row, np.in1d(ct_nm, self.features_names)]

		return features_matrix

	def content_ft(self, dataset):
		"""
		Create a feature matrix from the content of a review
		"""

		names_array = np.array([])
		features_matrix = np.array([])

		for n in range(1, self.ngram+1):

			print "Compute the " + str(n) + "-gram words"

			# Create TFIDF vectorizer for ngram words
			if n >= 4:
				tfidf_vec = text.TfidfVectorizer(tokenizer=analyze.get_text_ngram, ngram_range=(n, n))
			else:
				stop_words = text.ENGLISH_STOP_WORDS.union(["s", "t", "2", "m", "ve"])
				tfidf_vec = text.TfidfVectorizer(tokenizer=analyze.get_text_ngram, ngram_range=(n, n), stop_words=stop_words)

			# Compute the feature matrix
			tfidf = tfidf_vec.fit_transform([r.content for r in dataset])
			names = np.array(tfidf_vec.get_feature_names())

			# Sort and partition the tfidf matrix of ngram words
			tfidf_sum = np.array(tfidf.sum(axis=0).transpose().flatten())
			split_indexes = tfidf_sum.argsort()[0]
			low_b = int(self.low_boundary * len(split_indexes))
			up_b = low_b + int(self.frac * len(split_indexes))
			tfidf_sorted = tfidf[:, split_indexes[low_b:up_b]]
			names_sorted = names[split_indexes[low_b:up_b]]

			# Fill the feature matrix with the values
			if features_matrix.size == 0:
				features_matrix = tfidf_sorted
				names_array = names_sorted
			else:
				features_matrix = hstack((features_matrix, tfidf_sorted))
				names_array = np.hstack((names_array, names_sorted))

		return features_matrix, names_array

	def summary_ft(self):
		"""
		Create a feature matrix from the summary of a review
		"""

		return

	def product_ft(self):
		"""
		Create a feature matrix from the product of a review
		"""

		return

	def author_ft(self):
		"""
		Create a feature matrix from the author of a review
		"""

		return

	def date_ft(self):
		"""
		Create a feature matrix from the date of a review
		"""

		return

	def helpful_ft(self):
		"""
		Create a feature matrix from the helpfulness of a review
		"""

		return

	def get_feature_names(self):
		"""
		Return an array with the features name
		"""

		return self.features_names


def save_ft(ft, model, location=DEFAULT_FT_LOCATION):
	"""
	Save the features
	"""

	ts = utils.timestamp()
	filename = os.path.join(location, "ft_" + \
		ts + ".pkl")
	utils.dump_pickle(ft, filename)

	filename = os.path.join(location, "ft_model_" + \
		ts + ".pkl")
	utils.dump_pickle(model, filename)

	return filename



def main():
	"""
	Load data and create features
	"""

	dataset = data.load_pickled_data()
	training_set, validate_set = train_test_split(dataset["train"], test_size=VALIDATE_PART, random_state=RANDOM)
	training_set = training_set[0:10000]
	validate_set = validate_set
	ft_extractor = ReviewsFeaturesExtractor(ngram=1)

	training_ft = ft_extractor.fit_transform(training_set)
	validate_ft = ft_extractor.fit_transform(validate_set)

	print training_ft.shape, validate_ft.shape, ft_extractor.fit(np.array([]))

	save_ft(training_ft, ft_extractor)


if __name__ == '__main__':

	main()
