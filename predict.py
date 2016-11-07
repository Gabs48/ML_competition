""" Script that transform the content of train in sklearn boolean features """

import data
import utils
import indexation

from collections import defaultdict
import numpy as np
import os
from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import sys


DEFAULT_DATA_LOCATION = 'Data'
DEFAULT_FT_PATH = os.path.join(DEFAULT_DATA_LOCATION, 'features.pkl')
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_DATA_LOCATION, 'model.pkl')


def get_ft(path=DEFAULT_FT_PATH):
  """
  Load feature vectors
  """
  return utils.load_pickle(path)


def create_target(dataset):

	target = []
	for review in dataset:
		target.append(review.rating)

	return np.array(target)


def save_model(model, path=DEFAULT_MODEL_PATH):
  """
  Save the training model
  """
  utils.dump_pickle(model, path)


def create_model(ft, target):
  """
  Extract features in two steps:
   - Creation of a vect dict
   - PCA to reduce dimensionality
  """

  # Create a sklearn pipe
  clf = LogisticRegression()
  pipe = Pipeline([('classifier', clf)])

  # Apply pipe
  model = pipe.fit_transform(ft, target)

  # Save features
  save_model(model)

  return model

def main():
  data.create_pickled_data(overwrite_old=False)
  dataset = data.load_pickled_data()
  train_set = dataset['train']

  # Get features and target
  target = create_target(train_set)
  ft = get_ft()

  # Perform linear regression
  model = create_model(ft, target)

if __name__ == '__main__':
  main()
