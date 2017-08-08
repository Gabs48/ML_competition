"""Script loading the pickle files and creating a set of features to be processed."""

import analyze
import data
import utils

from itertools import izip
import numpy as np
import os
from scipy.sparse import hstack, csr_matrix, lil_matrix, coo_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer, text
from sklearn.model_selection import *
import sys
import time

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

	def __init__(self, max_df=0.2, min_df=0.001, ngram=4):

		self.max_df = float(max_df)
		self.min_df = float(min_df)
		self.ngram = ngram
		self.features_names = np.array([])
		self.dataset = None
		self.content_fct = None

	def fit(self, x, y=None):

		return self

	def fit_transform(self, dataset):
		"""
		Extract and combine all features
		"""

		# Extract feature matrices

		stop_words = text.ENGLISH_STOP_WORDS.union(["s", "t", "2", "m", "ve"])
		self.content_fct = text.TfidfVectorizer(tokenizer=analyze.get_text_ngram, ngram_range=(1, self.ngram), \
			min_df=self.min_df, max_df=self.max_df, stop_words=stop_words)

		tfidf_features = self.content_fct.fit_transform([r.content for r in dataset])
		names = np.array(self.content_fct.get_feature_names())

		return tfidf_features

		# print "Fill the features matrices"
		#
		# # Training phase
		# if self.features_names.size == 0:
		# 	ct_ft, ct_nm = self.content_ft(dataset)
		# 	features_matrix = ct_ft
		# 	self.features_names = ct_nm
		#
		# # Test and validation phase
		# else:
		#
		# 	t_it = time.time()
		# 	ct_ft, ct_nm = self.content_ft(dataset)
		# 	print time.time() - t_it
		# 	t_it = time.time()
		# 	#ct_ft = ct_ft.tocoo()
		# 	cols = []
		# 	rows = []
		# 	datas = []
		# 	#print type(ct_ft), ct_ft.nnz, len(ct_ft.data), ct_ft.shape, len( zip(ct_ft.row, ct_ft.col, ct_ft.data))
		# 	a = 0
		# 	features_matrix = csr_matrix((len(dataset), self.features_names.size))
		#
		# 	for i, n in enumerate(ct_nm):
		# 		indexes = np.where(self.features_names == n)[0]
		# 		if indexes.size != 0:
		# 			old_names_idx = np.where(ct_nm.indices == indexes[0])
		# 			print old_names_idx
		# 	new_names_idx = np.where(mat_csr.indices == b)
		#
		# 	features_matrix.indices[old_names_idx] = a
		# 	print
		# 	return mat_csr.asformat(mat.format)

			# for i, row in enumerate(dataset):
			#	print np.where(np.in1d(ct_nm, self.features_names))[0], ct_nm.size, self.features_names.size
			#	features_matrix[row, np.in1d(self.features_names, ct_nm)] = ct_ft.tolil()[row, np.in1d(ct_nm, self.features_names)]
			#	#print i#, len(np.in1d(self.features_names, ct_nm)), len(self.features_names)
			# print time.time() - t_it

			#for i, n in enumerate(ct_nm):
			#		indexes = np.where(self.features_names == n)[0]
			#		if indexes.size != 0:
			#			column_sparse = ct_ft[:, i].tocoo()
			#			for row, data in zip(column_sparse.row, column_sparse.data):
			#				cols.append(indexes[0])
			#				rows.append(row)
			#				datas.append(data)
			#
			#			print time.time() - t_it


			#for i, j, v in zip(ct_ft.row, ct_ft.col, ct_ft.data):
			#		if a%100:
			#				print a
			#				rows.append(i)
			#				cols.append(np.where( self.features_names==ct_nm[j] ))
			#			a += 1
			# features_matrix = lil_matrix(( ct_ft.data, (ct_ft.row, cols)), shape=(ct_ft.shape[0], self.features_names.size))
			# print ct_ft.tocsr().shape, features_matrix.shape, np.in1d(self.features_names, ct_nm).shape, np.in1d(ct_nm, self.features_names).shape
			#for row in range(features_matrix.shape[0]):
			# features_matrix[:, np.in1d(self.features_names, ct_nm)] = ct_ft.tolil()[:, np.in1d(ct_nm, self.features_names)]


		# return features_matrix

	def transform(self, dataset):
		print "Process the text"

		tfidf_features = self.content_fct.transform([r.content for r in dataset])
		names = np.array(self.content_fct.get_feature_names())

		return tfidf_features


	def sort_coo(matrix, names_origin, names_fin):
		tuples = izip(m.row, m.col, m.data)
		return sorted(tuples, key=lambda x: (x[0], x[2]))

	def content_ft(self, dataset):
		"""
		Create a feature matrix from the content of a review
		"""

		#names_array = np.array([])
		#features_matrix = np.array([])




		# for n in range(1, self.ngram+1):
		#
		# 	#print "Compute the " + str(n) + "-gram words"
		#
		# 	# Create TFIDF vectorizer for ngram words
		# 	if n >= 4:
		# 		tfidf_vec = text.TfidfVectorizer(tokenizer=analyze.get_text_ngram, ngram_range=(n, n))
		# 	else:
		# 		stop_words = text.ENGLISH_STOP_WORDS.union(["s", "t", "2", "m", "ve"])
		# 		tfidf_vec = text.TfidfVectorizer(tokenizer=analyze.get_text_ngram, ngram_range=(n, n), stop_words=stop_words)
		#
		# 	# Compute the feature matrix
		# 	tfidf = tfidf_vec.fit_transform([r.content for r in dataset])
		# 	names = np.array(tfidf_vec.get_feature_names())
		#
		# 	# Sort and partition the tfidf matrix of ngram words
		# 	tfidf_sum = np.array(tfidf.sum(axis=0).transpose().flatten())
		# 	split_indexes = tfidf_sum.argsort()[0]
		# 	low_b = int(self.low_boundary * len(split_indexes))
		# 	up_b = low_b + int(self.frac * len(split_indexes))
		# 	tfidf_sorted = tfidf[:, split_indexes[low_b:up_b]]
		# 	names_sorted = names[split_indexes[low_b:up_b]]
		#
		# 	# Fill the feature matrix with the values
		# 	if features_matrix.size == 0:
		# 		features_matrix = tfidf_sorted
		# 		names_array = names_sorted
		# 	else:
		# 		features_matrix = hstack((features_matrix, tfidf_sorted))
		# 		names_array = np.hstack((names_array, names_sorted))

		return tfidf, names

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
	ft_extractor = ReviewsFeaturesExtractor(ngram=3)

	training_ft = ft_extractor.fit_transform(training_set)
	validate_ft = ft_extractor.transform(validate_set)

	print training_ft.shape, validate_ft.shape, ft_extractor.fit(np.array([]))

	save_ft(training_ft, ft_extractor)


if __name__ == '__main__':

	main()
