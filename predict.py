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


def _parse_args(args):
  """
  Parse argument for validation
  """

  if not len(args) in (1, 2):
    print ('Usage: python2 predict.py <path_to_model.pkl> ')
    return

  path = sys.argv[1] if len(sys.argv) == 2 else DEFAULT_FT_PATH
  
  return path
  


def main(path):
  dataset = data.load_pickled_data()
  train_data = dataset['train']
  test_data = dataset['test']

  predictions_avg = predict_average(train_data, test_data)
  pred_file_name = utils.generate_unqiue_file_name(
      AVG_PREDICTIONS_BASENAME, 'npy')
  utils.dump_npy(predictions_avg, pred_file_name)
  print 'Dumped predictions to %s' % pred_file_name



if __name__ == '__main__':
  args = sys.argv
  arguments = _parse_args(args)
  if arguments:
    main(*arguments)