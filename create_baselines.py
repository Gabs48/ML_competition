"""Sample script creating some baseline predictions."""

import os
import numpy as np

import data
import utils


ALL_ZERO_PREDICTIONS_BASENAME = os.path.join('Predictions', 'all_zero')
AVG_PREDICTIONS_BASENAME = os.path.join('Predictions', 'average')

def predict_average(train_data, test_data):
  targets = np.array([review.rating for review in train_data])
  avg = targets.mean()
  predictions = [avg] * len(test_data)
  return predictions


def predict_zeros(test_data):
  predictions = [0.] * len(test_data)
  return predictions


def main():
  dataset = data.load_pickled_data()
  train_data = dataset['train']
  test_data = dataset['test']

  predictions_zero = predict_zeros(test_data)
  pred_file_name = utils.generate_unqiue_file_name(
      ALL_ZERO_PREDICTIONS_BASENAME, 'npy')
  utils.dump_npy(predictions_zero, pred_file_name)
  print 'Dumped predictions to %s' % pred_file_name

  predictions_avg = predict_average(train_data, test_data)
  pred_file_name = utils.generate_unqiue_file_name(
      AVG_PREDICTIONS_BASENAME, 'npy')
  utils.dump_npy(predictions_avg, pred_file_name)
  print 'Dumped predictions to %s' % pred_file_name



if __name__ == '__main__':
  main()