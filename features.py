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
		self.content_fct = None

	def fit(self, x, y=None):

		return self

	def fit_transform(self, dataset):
		"""
		Compute a new set of features from all dataset fields
		"""

		# Extract feature matrices

		stop_words = text.ENGLISH_STOP_WORDS.union(["s", "t", "2", "m", "ve"])
		self.content_fct = text.TfidfVectorizer(tokenizer=analyze.get_text_ngram, ngram_range=(1, self.ngram), \
			min_df=self.min_df, max_df=self.max_df, stop_words=stop_words)

		tfidf_features = self.content_fct.fit_transform([r.content for r in dataset])
		self.features_names = np.array(self.content_fct.get_feature_names())

		return tfidf_features

	def transform(self, dataset):
		"""
		Compute a set of features for a training dataset according to what has been trained previously
		"""

		tfidf_features = self.content_fct.transform([r.content for r in dataset])

		return tfidf_features


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
