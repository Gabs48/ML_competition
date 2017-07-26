"""Module for loading the train and test data.

This module supports loading the data in two seperate ways.
The first, which is done by calling the load_train and load_test functions,
opens and parses the text files one by one. Because of the abundance of files,
this can take a while.
The second way of loading the data is by opening a pickle file containing the
results of the load_train and load_test functions. To create this pickle file,
call the create_data_pickle function once. Afterwards, you will be able to
quickly load the data using the load_data_pickle function.
"""
import collections
import glob
import os
import re
import warnings

import utils


## Default data folder names ##


DEFAULT_DATA_LOCATION = 'Data'
DEFAULT_TRAIN_DATA_LOCATION = os.path.join(DEFAULT_DATA_LOCATION, 'Train')
DEFAULT_TEST_DATA_LOCATION = os.path.join(DEFAULT_DATA_LOCATION, 'Test')

DEFAULT_PICKLE_PATH = os.path.join(DEFAULT_DATA_LOCATION, 'data.pkl')

PRODUCT_REVIEWS_FILE_TEMPLATE = '*.txt'
TEST_REVIEWS_FILE_TEMPLATE = '*.txt'


## Class to store the parsed data in. ##

Review = collections.namedtuple(
  'Review', [
      'id', 'author', 'product', 'date',
      'summary', 'content', 'helpful', 'rating'])


## Helper functions for parsing data ##


def _extract_id_from_file_name(review_file_name):
  regex_res = re.findall('\d+', review_file_name)
  assert len(regex_res) == 1
  return int(regex_res[0])


_AUTHOR_TAG = '<Author>'
_PRODUCT_TAG = '<Product>'
_DATE_TAG = '<Date>'
_SUMMARY_TAG = '<Summary>'
_CONTENT_TAG = '<Content>'
_HELPFUL_TAG = '<Helpful>'
_RATING_TAG = '<Rating>'


_NR_LINES_REVIEW_TRAIN = 7
_NR_LINES_REVIEW_TEST = 5


def _parse_string_line(line, tag):
  assert line.startswith(tag)
  return line.strip()[len(tag):]


def _parse_author_line(author_line):
  return _parse_string_line(author_line, _AUTHOR_TAG)


def _parse_product_line(product_line):
  return _parse_string_line(product_line, _PRODUCT_TAG)


def _parse_date_line(date_line):
  return _parse_string_line(date_line, _DATE_TAG)


def _parse_summary_line(summary_line):
  return _parse_string_line(summary_line, _SUMMARY_TAG)


def _parse_content_line(content_line):
  return _parse_string_line(content_line, _CONTENT_TAG)


def _parse_helpful_line(helpful_line):
  helpful_string = _parse_string_line(helpful_line, _HELPFUL_TAG)
  positive, total = helpful_string.split('/')
  helpful_tuple = int(positive), int(total)
  assert int(positive) <= int(total)
  return helpful_tuple


def _parse_rating_line(rating_line):
  rating_string = _parse_string_line(rating_line, _RATING_TAG)
  return int(rating_string)



def _parse_single_product_review(review_lines):
  (author_line, product_line, date_line, summary_line,
      content_line, helpful_line, rating_line) = review_lines
  # Parse each of the lines
  author = _parse_author_line(author_line)
  product = _parse_product_line(product_line)
  date = _parse_date_line(date_line)
  summary = _parse_summary_line(summary_line)
  content = _parse_content_line(content_line)
  helpful = _parse_helpful_line(helpful_line)
  rating = _parse_rating_line(rating_line)
  return Review(
      id=-1, author=author, product=product, date=date, summary=summary,
      content=content, helpful=helpful, rating=rating)


def _parse_product_review_file(product_review_lines):
  assert (len(product_review_lines)+1)%(_NR_LINES_REVIEW_TRAIN+1) == 0
  product_reviews = []
  for start_idx in xrange(0, len(product_review_lines), (_NR_LINES_REVIEW_TRAIN+1)):
    review_lines = product_review_lines[
        start_idx : start_idx+_NR_LINES_REVIEW_TRAIN]
    product_reviews.append(_parse_single_product_review(review_lines))
  return product_reviews


def _parse_test_review_file(test_review_lines, review_id):
  assert len(test_review_lines) == _NR_LINES_REVIEW_TEST
  author_line, product_line, date_line, summary_line, content_line = (
      test_review_lines)
  author = _parse_author_line(author_line)
  product = _parse_product_line(product_line)
  date = _parse_date_line(date_line)
  summary = _parse_summary_line(summary_line)
  content = _parse_content_line(content_line)
  review = Review(
      id=review_id, author=author, product=product, date=date,
      summary=summary, content=content, rating=-1, helpful=(-1, -1))
  return review


## Functions for loading and parsing all of the data ##


def load_train(train_data_folder=DEFAULT_TRAIN_DATA_LOCATION):
  """Loads and parses the train data.

  Args:
    train_data_folder: string containing the path to the folder containing the
        training data.

  Returns:
    A list of all the reviews. Each review is a namedtuple object containing
    the author, product, date, summary, content, rating, and helpfulnesss data.
    The helpfulness is a tupleof two integers indicating how many people found
    the review helpful, out of the total amount of votes.
  """
  review_file_paths = glob.glob(os.path.join(
      train_data_folder, PRODUCT_REVIEWS_FILE_TEMPLATE))

  # load and parse each of the review files
  reviews = []
  for review_file_path in review_file_paths:
    with open(review_file_path, 'rb') as review_file:
      review_file_lines = review_file.readlines()
    product_reviews = _parse_product_review_file(review_file_lines)
    reviews += product_reviews
  return reviews


def load_test(test_data_folder=DEFAULT_TEST_DATA_LOCATION):
  """Loads and parses the test data.

  Args:
    test_data_folder: string containing the path to the folder containing the]
        test data.

  Returns:
    A list of all the test reviews, sorted by the review id. Each review is a
    namedtuple object containing the author, product, date, summary and content
    data. The rating and helpfulness fields are not available for test data.
  """
  review_file_paths = glob.glob(os.path.join(
      test_data_folder, TEST_REVIEWS_FILE_TEMPLATE))
  # Parse all the review files one by one
  reviews = []
  for review_file_path in review_file_paths:
    review_id = _extract_id_from_file_name(os.path.basename(review_file_path))
    with open(review_file_path, 'rb') as review_file:
      review_file_lines = review_file.readlines()
    reviews.append(_parse_test_review_file(review_file_lines, review_id))
  # Sort them by review id
  key_getter = lambda r: r.id
  reviews.sort(key=key_getter)
  assert range(1, len(reviews)+1) == map(key_getter, reviews)
  return reviews


## Functions for loading the data from a pickle file. ##


def create_pickled_data(train_data_folder=DEFAULT_TRAIN_DATA_LOCATION,
                        test_data_folder=DEFAULT_TEST_DATA_LOCATION,
                        pickled_data_file_path=DEFAULT_PICKLE_PATH,
                        overwrite_old=True):
  """Creates the data pickle file.

  Loads and parses the train and test data, and then writes it to a single
  pickle file.

  Args:
    train_data_folder: path to the train data folder.
    test_data_folder: path to the test data folder.
    pickled_data_file_path: location where the resulting pickle file should
        be stored.
  """
  if os.path.exists(pickled_data_file_path):
    if not overwrite_old:
      return
    warnings.warn(
        "There already exists a data pickle file, which will be overwritten.")
  train_data = load_train(train_data_folder)
  test_data = load_test(test_data_folder)
  utils.dump_pickle(
      dict(train=train_data, test=test_data), pickled_data_file_path)


def load_pickled_data(pickled_data_file_path=DEFAULT_PICKLE_PATH):
  """Loads the train and test data from a pickle file.

  Args:
    pickled_data_file_path: location of the data pickle file.
  """
  return utils.load_pickle(pickled_data_file_path)
