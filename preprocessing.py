"""All data and feature processing functions and routines"""

__author__ = "Gabriel Urbain"
__copyright__ = "Copyright 2017, Gabriel Urbain"

__license__ = "MIT"
__version__ = "0.2"
__maintainer__ = "Gabriel Urbain"
__email__ = "gabriel.urbain@ugent.be"
__status__ = "Research"
__date__ = "September 1st, 2017"

import analysis
import data
import utils

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import numpy as np
import sys

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import text, DictVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler

reload(sys)  
sys.setdefaultencoding('utf8')

DEFAULT_FT_LOCATION = 'Features'
TRAINING_PART = 0.95
VALIDATE_PART = 1 - TRAINING_PART
RANDOM = 42
SEED = 659


class ItemSelector(BaseEstimator, TransformerMixin):
    """
    This class selects elements by key in a dataset to feed specific estimators
    """

    def __init__(self, key, typ="list"):

        assert (typ in ["list", "dict"]), "Item can only return list of dict types!"
        assert (key in ["content", "product", "author"]), "Available keys are content, product, author!"
        self.key = key
        self.typ = typ

    def fit(self, x, y=None):

        return self

    def transform(self, dataset):

        if self.typ == "list":
            liste = []
            if self.key == "content":
                liste = [r.content for r in dataset]
            elif self.key == "author":
                liste = [r.author for r in dataset]
            elif self.key == "product":
                liste = [r.product for r in dataset]
            return liste

        elif self.typ == "dict":
            dictionary = []
            if self.key == "content":
                dictionary = [{r.content: 1} for r in dataset]
            elif self.key == "author":
                dictionary = [{r.author: 1} for r in dataset]
            elif self.key == "product":
                dictionary = [{r.product: 1} for r in dataset]
            return dictionary

        else:
            print "Type error in ItemSelector!"
            return


class Float2Labels(BaseEstimator, TransformerMixin):

    def __init__(self, min_r=0, max_r=5):

        self.min_r = min_r
        self.max_r = max_r

    def transform(self, X, *_):

        # Threshold
        lv = X < self.min_r
        hv = X > self.max_r
        X[lv] = self.min_r
        X[hv] = self.max_r

        return np.rint(X)

    def fit(self, *_):

        return self


def loss_fct(truth, prediction):
    """
    Evaluate the gap between the target and prediction
    """

    diff = truth - prediction
    score = float(np.sum(np.abs(diff))) / diff.size

    return score


def create_target(dataset):

    target = []
    for r in dataset:
        target.append(r.rating)

    return np.array(target)


def create_ft_ct_pd_au(ngram=3, max_df=0.3, min_df=0.0003, w_ct=1, w_pd=1, w_au=1):

    # Declare estimators
    stop_words = text.ENGLISH_STOP_WORDS.union(["s", "t", "2", "m", "ve"])
    content_fct = text.TfidfVectorizer(tokenizer=analysis.get_text_ngram, ngram_range=(1, ngram),
                                       min_df=min_df, max_df=max_df, stop_words=stop_words)
    product_fct = DictVectorizer()
    author_fct = DictVectorizer()

    # Create features pipe
    tl = [('content', Pipeline([
                                ('selector_ct', ItemSelector(key='content')),
                                ('content_ft', content_fct),
                                ])),
          ('product', Pipeline([
                                ('selector_pd', ItemSelector(key='product', typ="dict")),
                                ('product_ft', product_fct),
                                ])),
          ('author', Pipeline([
                                ('selector_au', ItemSelector(key='author', typ="dict")),
                                ('author_ft', author_fct),
                                ]))
          ]
    tw = {'content': w_ct, 'product': w_pd, 'author': w_au}

    return Pipeline([('ft_extractor', FeatureUnion(transformer_list=tl, transformer_weights=tw))])


def create_ft_ctsvd_pd_au(ngram=3, k=10000, max_df=0.3, min_df=0.0003, w_ct=1, w_pd=1, w_au=1):

    # Declare estimators
    stop_words = text.ENGLISH_STOP_WORDS.union(["s", "t", "2", "m", "ve"])
    content_fct = text.TfidfVectorizer(tokenizer=analysis.get_text_ngram, ngram_range=(1, ngram),
                                       min_df=min_df, max_df=max_df, stop_words=stop_words)
    product_fct = DictVectorizer()
    author_fct = DictVectorizer()

    # Create features pipe
    tl = [('content', Pipeline([
                                ('selector_ct', ItemSelector(key='content')),
                                ('ft', content_fct),
                                ('reductor', TruncatedSVD(n_components=int(k))),
                                ])),
          ('product', Pipeline([
                                ('selector_pd', ItemSelector(key='product', typ="dict")),
                                ('ft', product_fct),
                                ])),
          ('author', Pipeline([
                                ('selector_au', ItemSelector(key='author', typ="dict")),
                                ('ft', author_fct),
                                ]))
          ]
    tw = {'content': w_ct, 'product': w_pd, 'author': w_au}

    return Pipeline([('ft_extractor', FeatureUnion(transformer_list=tl, transformer_weights=tw))])


def create_ft_ct(ngram=3, max_df=0.3, min_df=0.0003):

    stop_words = text.ENGLISH_STOP_WORDS.union(["s", "t", "2", "m", "ve"])
    content_fct = text.TfidfVectorizer(tokenizer=analysis.get_text_ngram, ngram_range=(1, ngram),
                                       min_df=min_df, max_df=max_df, stop_words=stop_words)

    return Pipeline(([('selector_ct', ItemSelector(key='content')),	('content_ft', content_fct)]))


def save_ft(test_ft, train_ft, target, filename=None):

    ts = utils.timestamp()
    if filename is None:
        filename = os.path.join(DEFAULT_FT_LOCATION, "ft_" + ts + ".pkl")
    utils.dump_pickle((test_ft, train_ft, target), filename)

    return filename


def load_ft(filename):

    (test_ft, train_ft, target) = utils.load_pickle(filename)

    return test_ft, train_ft, target


def create_ft(svd=True):

    # 1. Get the data
    print "1. Load data\n"
    train_set = data.load_pickled_data()['train']
    test_set = data.load_pickled_data()['test']
    target = create_target(train_set)

    # 2. Create the feature matrices
    print "2. Create features"
    if svd:
        ft_pipe = Pipeline([('ft', create_ft_ct_pd_au()), ('red', TruncatedSVD(n_components=1000)),
                            ('norm', MinMaxScaler())])
    else:
        ft_pipe = Pipeline([('ft', create_ft_ct_pd_au()), ('norm', MaxAbsScaler())])

    train_ft = ft_pipe.fit_transform(train_set)
    test_ft = ft_pipe.transform(test_set)
    print "Train features matrix size: " + str(train_ft.shape) + " and target size: " + str(len(target))
    print "Test features matrix size: " + str(test_ft.shape) + "\n"

    # 3. Save features
    print "3. Save features"
    if svd:
        save_ft(test_ft, train_ft, target, filename=DEFAULT_FT_LOCATION + "/ft_svd.pkl")
        r, c = train_ft.nonzero()
        feature_array = train_ft[r, c].flatten().tolist()
        plt.hist(feature_array, 10, alpha=0.75)
        plt.title('Features Histogram')
        plt.tight_layout()
        plt.savefig(DEFAULT_FT_LOCATION + "/histogram_svd.png", format='png', dpi=300)
        plt.close()
    else:
        save_ft(test_ft, train_ft, target, filename=DEFAULT_FT_LOCATION + "/ft_max_scaler.pkl")
        r, c = train_ft.nonzero()
        feature_array = train_ft[r, c].flatten().tolist()
        plt.hist(feature_array, 50, alpha=0.75)
        plt.title('Features Histogram')
        plt.tight_layout()
        plt.savefig(DEFAULT_FT_LOCATION + "/histogram_max_scaler.png", format='png', dpi=300)
        plt.close()


if __name__ == '__main__':

    args = sys.argv

    if args[1] == "ft":
        create_ft()
    else:
        print "Option does not exist. Please, check the preprocessing.py file"
