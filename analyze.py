"""Script loading the data and analyse some of its properties."""

import data
from utils import *

from collections import Counter, OrderedDict
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.mlab import *
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import re
import string
import sys
import unicodedata

#plt.style.use('fivethirtyeight')

reload(sys)  
sys.setdefaultencoding('utf8')

DEFAULT_AN_LOCATION = 'Analysis'

def to_percent(y, position):
	# Ignore the passed in position. This has the effect of scaling the default
	# tick locations.
	s = str(100 * y)

	# The percent symbol needs escaping in latex
	if matplotlib.rcParams['text.usetex'] is True:
		return s + r'$\%$'
	else:
		return s + '%'


def seg_norm_text(text):
	"""
	Segment the text according to points and comma, and
	normalize it by removing all caps, accent and special char
	"""
 
 	words = [x.strip() for x in re.split('[.,\/#!$%\^&\*;:{}=\-_`~()]', text)]
	
	for i, t in enumerate(words):
		words[i] = str(''.join(x for x in unicodedata.normalize('NFKD', unicode(t)) \
		if x in (string.ascii_letters + string.whitespace)).lower()).split(" ")

	return words

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


def plot_rating_distro(trainset):
	"""
	Bar plot for the score distribution
	"""

	ratings = np.array([r.rating for r in trainset])

	d = np.diff(np.unique(ratings)).min()
	left_of_first_bin = ratings.min() - float(d)/2
	right_of_last_bin = ratings.max() + float(d)/2
	n, bins, patches = plt.hist(ratings, np.arange(left_of_first_bin, right_of_last_bin + d, d), normed=True)
	formatter = FuncFormatter(to_percent)
	plt.gca().yaxis.set_major_formatter(formatter)

	plt.savefig(DEFAULT_AN_LOCATION + "/rating_distro.png", format='png', dpi=300)
	plt.close()


def plot_content_distro(trainset, forceReprocess=False):
	"""
	Bar plot for the word distribution of the review content
	"""

	MAXPLOT = 200
	size = len(trainset)

	for n_shingle in [3]:
		filename = os.path.join(DEFAULT_AN_LOCATION, "shingle_" + str(n_shingle) + ".pkl")
		if os.path.exists(filename) and not forceReprocess:
			shingle_cnt = load_pickle(filename)
			print 'Shingle ' + str(n_shingle) + ' loaded from file!'

		else:
			shingle = []
			i = 0
			for r in trainset:
				if i%1000 == 0:
					print 'Shingle ' + str(n_shingle) + ' processing: ' + n2str(float(i)/size*100) + ' %'
				s = create_review_shingle(r, n=n_shingle)
				shingle.extend(s)
				i += 1

			shingle_cnt = Counter(shingle)
			dump_pickle(shingle_cnt, filename)

		shingle_most_common = OrderedDict(shingle_cnt.most_common(MAXPLOT))

		labels, values = zip(*shingle_most_common.items())
		indexes = np.arange(len(labels))
		width = 1
		plt.bar(indexes, values, width)
		plt.savefig(DEFAULT_AN_LOCATION + "/shingle_" + str(n_shingle) + "_distro.png", format='png', dpi=300)
		plt.close()


def main():
	"""
	Load data and create features
	"""

	dataset = data.load_pickled_data()
	train_set = dataset['train']
	make_dir(DEFAULT_AN_LOCATION)

	plot_rating_distro(train_set)
	plot_content_distro(train_set)


if __name__ == '__main__':
	main() 
