""" Script that transform the content of train in sklearn boolean features """

import data
import features
import train
import utils

import numpy as np
import os
from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import sys


DEFAULT_FT_LOCATION = 'Features'
DEFAULT_MODEL_LOCATION = 'Models'
DEFAULT_FT_PATH = os.path.join(DEFAULT_FT_LOCATION, 'features.pkl')
DEFAULT_FT_NAMES_PATH = os.path.join(DEFAULT_FT_LOCATION, 'ft_names.pkl')
DEFAULT_FT_MODEL_PATH = os.path.join(DEFAULT_FT_LOCATION, 'ft_model.pkl')
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODEL_LOCATION, 'model.pkl')

def add_extra_test_ft(dataset, ft_dic, train_dict):
  """
  Add author and hotel id of review to ngram dictionnary
  TODO: homogeneize author, chotel and ngrams content before to mix them!!
  """

  print " -- Add author and hotel in features (to update) -- "
  for i, d in enumerate(ft_dic):
    hotel = "HOTEL " + str(dataset[i].hotel.id)
    auth = "AUTH " + str(dataset[i].author)
    if hotel in train_dict:
      d[hotel] = 1
    if auth in train_dict:
      d[auth] = 1

  return ft_dic


def get_test_ft(dataset, ft_mdl):
  """
  Compute the test features in the dataset given the set of
  trianing features
  """

  print ' -- Compute content features -- '
  train_dict = ft_mdl.steps[0][1].vocabulary_
  yes = 0
  no = 0

  ft_dic = []
  for i, review in enumerate(dataset):
    if i%500 == 0:
      print " -- Iteration " + str(i)
    d = dict()
    ngrams_list = list(set(features.create_review_shingle(review)))
    for ngram in ngrams_list:
      ngram_word = ' '.join(ngram)
      if ngram_word in train_dict:
        d[ngram_word] = 1
        yes += 1
      else:
        no += 1
    ft_dic.append(d)

  present = 100 * yes / float(yes + no)
  print " -- " + "{0:.2f}".format(present) + "% of the validation " + \
  "shingles were present in the feature model and are taken into account -- "
  ft_dic = add_extra_test_ft(dataset, ft_dic, train_dict)

  return ft_dic


def get_model(path=DEFAULT_MODEL_PATH):
  """
  Load feature vectors
  """

  return utils.load_pickle(path)


def get_ft_names(path=DEFAULT_FT_NAMES_PATH):
  """
  Load the list of features names
  """

  return utils.load_pickle(path)


def validate(val_set, ft_mdl_path=DEFAULT_FT_MODEL_PATH , lr_mdl_path=DEFAULT_MODEL_PATH):
  """
  Validate a model by returning the loss function on a validation set
  """

  print "\n -- VALIDATE A MODEL --"

  # Get test target
  val_tg = train.create_target(val_set)

  # Features extraction
  ft_mdl = get_model(path=ft_mdl_path)
  val_ft_dict = get_test_ft(val_set, ft_mdl)
  val_ft = ft_mdl.transform(val_ft_dict)

  # LR predcition
  lr_mdl = get_model(path=lr_mdl_path)
  prd = lr_mdl.predict(val_ft)
  score = train.loss_fct(val_tg, prd)

  # Evaluate and save
  print " -- Overal prediction Mean Absolute Error = " + str(score)
  train.save_score(score, lr_mdl_path, txt="validation")

  return score


def _parse_args(args):
  """
  Parse argument for validation
  """

  if not len(args) in (2, 3):
    print ('Usage: python2 validate.py <path_to_ft_model.plk> <path_to_lr_mdl.pkl> ')
    return

  ft_mdl_path = sys.argv[1]
  lr_mdl_path = sys.argv[2] if len(sys.argv) == 3 else DEFAULT_MODEL_PATH
  
  return ft_mdl_path, lr_mdl_path


def main(ft_mdl_path, lr_mdl_path):
  data.create_pickled_data(overwrite_old=False)
  dataset = data.load_pickled_data()#pickled_data_file_path='Data/data_short.pkl')
  val_size = int(np.ceil(features.VALIDATION_PART * len(dataset['train'])))
  val_set = dataset['train'][-val_size:]

  # Validation
  score = validate(val_set, ft_mdl_path=ft_mdl_path, lr_mdl_path=lr_mdl_path)
  

if __name__ == '__main__':
  args = sys.argv
  arguments = _parse_args(args)
  if arguments:
    main(*arguments)