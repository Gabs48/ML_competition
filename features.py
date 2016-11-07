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

DEFAULT_DATA_LOCATION = 'Data'
DEFAULT_FT_PATH = os.path.join(DEFAULT_DATA_LOCATION, 'features.pkl')

### DEPRECATED

from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import PCA

DEFAULT_INDEX_PATH = os.path.join(DEFAULT_DATA_LOCATION, 'index.npy')


def create_data_shingles(dataset):
  """ 
  Create and return sets of ngram words from the given field of a full dataset
  """

  shingles_list = []

  for i, r in enumerate(dataset):
    if i%2000 == 0:
      print " -- Process review " + str(i)
    shingles_list.extend(create_review_shingle(r))

  print " -- Shingle list size = " + str(sys.getsizeof(shingles_list)/1000000)  + " MB"
  shingles_set = set(shingles_list)
  print " -- Shingle set size = " + str(sys.getsizeof(shingles_set)/1000000) + " MB"
  return shingles_set


def save_shingles_index(dataset, index_file_path=DEFAULT_INDEX_PATH):
  """
  Create and save an index containing the different shingles
  """

  shingles_set = create_data_shingles(dataset)
  shingles_list = list(shingles_set)
  print " -- Shingle list size = " + str(sys.getsizeof(shingles_list)/1000000)  + " MB"
  index = np.array(shingles_list)
  print " -- Shingle index size = " + str(index.nbytes/1000000) + " MB"
  name = index_file_path #utils.generate_unqiue_file_name(index_file_path, 'npy')
  utils.dump_npy(index, name)


def ret_shingles_index_size(index_file_path=DEFAULT_INDEX_PATH):
  """
  Return the size of the index
  """

  return utils.load_npy(index_file_path)


def review_to_vec(index=None, index_file_path=DEFAULT_INDEX_PATH):
  """
  Transform reiview to a sparse binary vector based on an index table for a given field.
  The vector element is 1 if it exists in the index, 0 otherwise.
  """

  if index == None:
    index = utils.load_npy(index_file_path)
  vec = np.zeros(ret_shingles_index_size())
  return vec


### IN USE

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


def create_review_shingle(review, n=3, cat="content"):
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
  print " -- Convert review content to ngram dictionnary -- "
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


def save_ft(features, path=DEFAULT_FT_PATH):
  """
  Save the features
  """

  utils.dump_pickle(features, path)
  return


def create_ft(data):
  """
  Extract features in two steps:
   - Creation of a vect dict
   - PCA to reduce dimensionality (deprecated)
  """

  # Create a processing pipe
  vec = DictVectorizer()
  pipe = Pipeline([('vectorizer', vec)])

  # Apply pipe
  data_dict = create_data_dict(data)
  ft = pipe.fit_transform(data_dict)

  # Save features
  save_ft(ft)
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
