""" Script that transform the content of train in sklearn boolean features """

import data
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
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODEL_LOCATION, 'model.pkl')
VALIDATION_PART = 1 - train.TRAINING_PART


def get_model(path=DEFAULT_MODEL_PATH):
  """
  Load feature vectors
  """

  return utils.load_pickle(path)


def validate(val_set, ft_path=DEFAULT_FT_PATH , mdl_path=DEFAULT_MODEL_PATH):
  """
  Validate a model by returning the loss function on a validation set
  """

  print "\n -- VALIDATE A MODEL --"

  # Get model
  model = get_model(mdl_path)

  # Get test target
  val_tg = train.create_target(val_set)

  # Get test features
  val_ft = train.get_ft(mini=len(val_set), path=ft_path)

  print len(val_set), val_ft.shape

  # Predict
  predict = model.predict(val_ft)
  score = train.loss_fct(val_tg, predict)

  # Evaluate and save
  print " -- Overal prediction Mean Absolute Error = " + str(score)
  train.save_score(score, mdl_path, txt="validation")

  return score


def _parse_args(args):
  """
  Parse argument for validation
  """

  if not len(args) in (2, 3):
    print ('Usage: python2 validate.py <path_to_features.plk> <path_to_model.pkl> ')
    return

  ft_path = sys.argv[1]
  mdl_path = sys.argv[2] if len(sys.argv) == 3 else DEFAULT_MODEL_PATH
  
  return ft_path, mdl_path


def main(ft_path, mdl_path):
  data.create_pickled_data(overwrite_old=False)
  dataset = data.load_pickled_data()
  val_size = int(np.ceil(VALIDATION_PART * len(dataset['train'])))
  val_set = dataset['train'][-val_size:]

  # Validation
  score = validate(val_set, ft_path=ft_path, mdl_path=mdl_path)
  

if __name__ == '__main__':
  args = sys.argv
  arguments = _parse_args(args)
  if arguments:
    main(*arguments)