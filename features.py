"""Script loading the pickle files and creating a set of features to be processed."""

import analyze
import data
import utils
import train

import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import text, DictVectorizer
from sklearn.model_selection import *
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
import sys
import time

reload(sys)  
sys.setdefaultencoding('utf8')

DEFAULT_FT_LOCATION = 'Features'
TRAINING_PART = 0.95
VALIDATE_PART = 1 - TRAINING_PART
RANDOM = 42


class ItemSelector(BaseEstimator, TransformerMixin):
	"""
	This class selects elements by key in a dataset to feed specific estimators
	"""

	def __init__(self, key, typ="list"):

		assert (typ in ["list", "dict"]), "Item can only return list of dict types!"
		assert (key in ["content", "product", "author"]), "Available keys are content, product, author!"
		self.key = key
		self.typ = typ

	def fit(self, x, y=None):

		return self

	def transform(self, dataset):

		if self.typ == "list":
			liste = []
			if self.key == "content":
				liste = [r.content for r in dataset]
			elif self.key == "author":
				liste = [r.author for r in dataset]
			elif self.key == "product":
				liste = [r.product for r in dataset]
			return liste

		elif self.typ == "dict":
			dictionary = []
			if self.key == "content":
				dictionary = [{r.content: 1} for r in dataset]
			elif self.key == "author":
				dictionary = [{r.author: 1} for r in dataset]
			elif self.key == "product":
				dictionary = [{r.product: 1} for r in dataset]
			return dictionary

		else:
			print "Type error in ItemSelector!"
			return


def create_ft_ct_pd_au(ngram=3, max_df=0.3, min_df=0.0003, w_ct=1, w_pd=1, w_au=1):
	"""
	Create a feature extraction pipe given different hyper parameters and return it
	"""

	# Declare estimators
	stop_words = text.ENGLISH_STOP_WORDS.union(["s", "t", "2", "m", "ve"])
	content_fct = text.TfidfVectorizer(tokenizer=analyze.get_text_ngram, ngram_range=(1, ngram),
		min_df=min_df, max_df=max_df, stop_words=stop_words)
	product_fct = DictVectorizer()
	author_fct = DictVectorizer()

	# Create features pipe
	tl = [('content', Pipeline([
								('selector_ct', ItemSelector(key='content')),
								('content_ft', content_fct),
								])),
						('product', Pipeline([
								('selector_pd', ItemSelector(key='product', typ="dict")),
								('product_ft', product_fct),
								])),
						('author', Pipeline([
								('selector_au', ItemSelector(key='author', typ="dict")),
								('author_ft', author_fct),
								]))
						]
	tw = {'content': w_ct, 'product': w_pd, 'author': w_au}

	return Pipeline([('ft_extractor', FeatureUnion(transformer_list=tl, transformer_weights=tw))])


def create_ft_ct(ngram=3, max_df=0.3, min_df=0.0003):
	"""
	Create a feature extraction pipe using the review content only and return it
	"""

	stop_words = text.ENGLISH_STOP_WORDS.union(["s", "t", "2", "m", "ve"])
	content_fct = text.TfidfVectorizer(tokenizer=analyze.get_text_ngram, ngram_range=(1, ngram),
		min_df=min_df, max_df=max_df, stop_words=stop_words)

	return Pipeline(([('selector_ct', ItemSelector(key='content')),	('content_ft', content_fct)]))


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


def test_ft_lin_class(ft_type):
	"""
	Load data and test features extraction with a simple logistic regression classifier
	"""

	# Split the data
	dataset = data.load_pickled_data()
	training_set, validate_set = train_test_split(dataset["train"],
		test_size=VALIDATE_PART, random_state=356)
	training_set = training_set[0:20000]
	target_training = train.create_target(training_set)
	target_validate = train.create_target(validate_set)

	# Set up the model
	if ft_type == "content":
		ft_extractor = create_ft_ct(ngram=3, min_df=0.0001, max_df=0.3)
	elif ft_type == "content_product_author":
		ft_extractor = create_ft_ct_pd_au(ngram=3, min_df=0.0001, max_df=0.3, w_ct=1, w_pd=1, w_au=1)
	else:
		ft_extractor = create_ft_ct(ngram=3, min_df=0.0001, max_df=0.3)
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

	print (score_training, score_validate, t_training, t_validate)


def create_ft(ft_type):
	"""
	Load data and create features for the whole dataset
	"""

	# Get the data
	dataset = data.load_pickled_data()
	training_set = dataset["train"]

	# Set up the model
	if ft_type == "content":
		ft_extractor = create_ft_ct(ngram=3, min_df=0.0001, max_df=0.3)
	elif ft_type == "content_product_author":
		ft_extractor = create_ft_ct_pd_au(ngram=3, min_df=0.0001, max_df=0.3, w_ct=1, w_pd=1, w_au=1)
	else:
		ft_extractor = create_ft_ct(ngram=3, min_df=0.0001, max_df=0.3)

	# Train and save
	training_ft = ft_extractor.fit_transform(training_set)
	save_ft(training_ft, ft_extractor)


if __name__ == '__main__':

	args = sys.argv

	if args[1] == "test":
		test_ft_lin_class("content_product_author")
	elif args[1] == "compute":
		create_ft("content_product_author")
	else:
		print "Option does not exist. Please, check the features.py file"
