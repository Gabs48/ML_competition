""" Script that transform the content of train in sklearn boolean features """

import data
import utils

import numpy as np
import os
from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import sys


DEFAULT_MODEL_LOCATION = 'Models'
DEFAULT_FT_LOCATION = 'Features'
DEFAULT_FT_PATH = os.path.join(DEFAULT_FT_LOCATION, 'features.pkl')
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODEL_LOCATION, 'model.pkl')
TRAINING_PART = 0.95


def get_ft(mini=None, maxi=None, path=DEFAULT_FT_PATH):
  """
  Load feature vectors
  """
  ft = utils.load_pickle(path)
  if maxi == None:
    maxi = ft.shape[0]
  if mini == None:
    mini = 0
  
  return ft[-mini:maxi,:]


def create_target(dataset):

	target = []
	for review in dataset:
		target.append(review.rating)

	return np.array(target)


def save_model(model, location=DEFAULT_MODEL_LOCATION):
  """
  Save the training model
  """

  model_filename = os.path.join(location, "model_" + \
    utils.timestamp() + ".pkl")
  utils.dump_pickle(model, model_filename)

  return model_filename


def save_score(score, model_path=DEFAULT_MODEL_PATH, txt="train"):
  """
  Save score
  """

  filename = model_path.replace(".pkl", ".txt")
  filename = filename.replace("model_", txt + "_score_")

  # Save score
  file = open(filename, "w")
  file.write("Score: " + str(score))
  file.close()

  return


def loss_fct(target, prediction):
  """
  Evaluate the gap between the target and prediction
  """

  diff = target - prediction
  score = float(np.sum(np.abs(diff))) / diff.size

  return score


def create_model(ft, target, verbose=0):
  """
  Extract features in two steps:
   - Creation of a vect dict
   - PCA to reduce dimensionality
  """

  print "\n -- CREATE A LOGISTIC REGRESSION MODEL --"

  # Create a sklearn pipe
  clf = LogisticRegression(verbose=verbose)
  pipe = Pipeline([('classifier', clf)])

  # Apply pipe
  pipe.fit_transform(ft, target)

  # Verify
  predict = pipe.predict(ft)
  score = loss_fct(target, predict)

  # Save model and score
  model_filename = save_model(pipe)
  save_score(score, model_filename)
  print " -- Model created with Mean Absolute Error = " + \
    str(score) + " --"

  return pipe


def main(ft_path):
  data.create_pickled_data(overwrite_old=False)
  dataset = data.load_pickled_data()
  train_size = int(np.floor(TRAINING_PART * len(dataset['train'])))
  train_set = dataset['train'][0:train_size]

  # Get features and target
  target = create_target(train_set)
  ft = get_ft(maxi=train_size, path=ft_path)

  # Perform linear regression
  model = create_model(ft, target)


def _parse_args(args):
  """
  Parse argument for validation
  """

  if not len(args) in (1, 2):
    print ('Usage: python2 train.py <path_to_features.pkl> ')
    return

  ft_path = sys.argv[1] if len(sys.argv) == 2 else DEFAULT_FT_PATH
  
  return ft_path
  

if __name__ == '__main__':
  args = sys.argv
  ft_path = _parse_args(args)
  main(ft_path)