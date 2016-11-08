"""Script loading the pickle files and creating a set of features to be processed."""

import data
import utils

import numpy as np
import os
import string
import re
import unicodedata
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
import sys

reload(sys)  
sys.setdefaultencoding('utf8')

DEFAULT_FT_LOCATION = 'Features'


def seg_norm_text(data):
  """
  Segment the text according to points and comma, and
  normalize it by removing all caps, accent and special char
  """
 
  data_list = [x.strip() for x in re.split('[.,\/#!$%\^&\*;:{}=\-_`~()]', data)]
  
  for i, t in enumerate(data_list):
  	data_list[i] = str(''.join(x for x in unicodedata.normalize('NFKD', unicode(t)) \
  	if x in (string.ascii_letters + string.whitespace)).lower()).split(" ")

  return data_list


def create_review_shingle(review, n=2):
  """ 
  Create and return sets of ngram words from the given field of a review
  """

  r_list = seg_norm_text(review.content)
  shingles = []

  for r in r_list:
    if len(r) >= n:
        shingles.extend([tuple(r[i:i+n]) for i in xrange(len(r)-n)])

  return shingles
  

def create_data_dict(dataset):
  """
  Get a dict of features from the given set
  """
  # To dict
  print " -- Convert dataset content to ngram dictionnary -- "
  ft_dic = []
  for i, review in enumerate(dataset):
    if i%2000 == 0:
      print " -- Iteration " + str(i)
    d = dict()
    ngrams_list = list(set(create_review_shingle(review)))
    for ngram in ngrams_list:
      d[' '.join(ngram)] = 1
    ft_dic.append(d)

  return ft_dic


def add_extra_ft(dataset, data_list):
  """
  Add author and hotel id of review to ngram dictionnary
  TODO: homogeneize author, chotel and ngrams content before to mix them!!
  """

  print " -- Add author and hotel in features (to update) -- "
  for i, d in enumerate(data_list):
    hotel = "HOTEL " + str(dataset[i].hotel.id)
    auth = "AUTH " + str(dataset[i].author)
    d[hotel] = 1
    d[auth] = 1

  return data_list

def save_ft(features, location=DEFAULT_FT_LOCATION):
  """
  Save the features
  """

  filename = os.path.join(location, "features_" + \
    utils.timestamp() + ".pkl")
  utils.dump_pickle(features, filename)

  return filename


def save_pp_model(model, path):
  """
  Save the feature extraction model
  """
  utils.dump_pickle(model, path)


def create_ft(data):
  """
  Extract features in two steps:
   - Creation of a vect dict
   - PCA to reduce dimensionality (deprecated)
  """

  print "\n -- CREATE FEATURES MATRIX --"

  # Create content data dictionnary
  data_list_dict = create_data_dict(data)

  # Add special entries for author and hotel id
  data_list_dict = add_extra_ft(data, data_list_dict)
  
  # Create and execute a processing pipe for review content
  vec = DictVectorizer()
  pipe = Pipeline([('vectorizer', vec)])
  ft = pipe.fit_transform(data_list_dict)

  # Save features and preprocessing model
  save_ft(ft)
  #save_pp_model(pipe)

  return ft


def main():
  """
  Load data and create features
  """

  data.create_pickled_data(overwrite_old=False)
  dataset = data.load_pickled_data()
  train_set = dataset['train']
  create_ft(train_set)


if __name__ == '__main__':
  main()