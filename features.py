"""Script loading the pickle files and creating a set of features to be processed."""

import data
import utils

import numpy as np
import os
import string
import re
import unicodedata
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

import sys

reload(sys)  
sys.setdefaultencoding('utf8')

DEFAULT_FT_LOCATION = 'Features'
TRAINING_PART = 0.95
VALIDATION_PART = 1 - TRAINING_PART


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


def create_review_shingle(review, n=3):
  """ 
  Create and return sets of ngram words from the given field of a review
  """

  r_list = seg_norm_text(review.content)
  shingles = []

  for r in r_list:
    if len(r) >= n:
        for j in range(n):
          shingles.extend([tuple(r[i:i+n-j]) for i in xrange(len(r)-n)])

  return shingles
  

def create_data_dict(dataset):
  """
  Get a dict of features from the given set
  """

  print " -- Convert dataset content to ngram dictionnary -- "
  ft_dic = []
  for i, review in enumerate(dataset):
    if i%2000 == 0:
      print " -- Iteration " + str(i)
    d = dict()
    ngrams_list = list(set(create_review_shingle(review)))
    for ngram in ngrams_list:
      ngram_word = ' '.join(ngram)
      d[ngram_word] = 1
    ft_dic.append(d)

  return ft_dic


def add_extra_ft(dataset, ft_dic):
  """
  Add author and hotel id of review to ngram dictionnary
  TODO: homogeneize author, chotel and ngrams content before to mix them!!
  """

  print " -- Add author and hotel in features (to update) -- "
  for i, d in enumerate(ft_dic):
    hotel = "HOTEL " + str(dataset[i].hotel.id)
    auth = "AUTH " + str(dataset[i].author)
    d[hotel] = 1
    d[auth] = 1

  return ft_dic


def create_list_content(dataset):

  rev = []
  for r in dataset:
    rev.append(r.content)

  return rev


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


def create_ft(data):
  """
  Extract features and create a vectorized dictionnary
  """

  print "\n -- CREATE FEATURES MATRIX --"

  # Create content data dictionnary
  ft_dic = create_data_dict(data)

  # Add special entries for author and hotel id
  ft_dic = add_extra_ft(data, ft_dic)

  # Create a list of content if we don't preprocess it manually
  # rev = create_list_content(data)

  # Create and execute a processing pipe for review content
  cv = CountVectorizer()
  dv = DictVectorizer()
  tfidfv = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=20000)
  tfidft = TfidfTransformer()
  pca = TruncatedSVD(n_components=1000)

  model = Pipeline([("vec", dv), ('tfidf', tfidfv)])#, ("sel", selecKBest)])#, ('pca', pca)])
  ft = model.fit_transform(ft_dic)
  print ft

  return ft, model


def main():
  """
  Load data and create features
  """

  dataset = data.load_pickled_data()#pickled_data_file_path='Data/data_short.pkl')
  train_size = int(np.floor(TRAINING_PART * len(dataset['train'])))
  train_set = dataset['train'][0:train_size]
  ft, ft_model = create_ft(train_set)

  # Save features, names and preprocessing model
  save_ft(ft, ft_model)

if __name__ == '__main__':
  main()