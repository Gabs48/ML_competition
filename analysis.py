"""Script loading the data and analyse some of its properties."""

import data
import utils

import datetime
from collections import Counter
import matplotlib
matplotlib.use("Agg")
from matplotlib.dates import DateFormatter
from matplotlib.mlab import *
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import sys

from nltk.tokenize import RegexpTokenizer
from nltk.util import ngrams
from nltk import wordpunct_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text

reload(sys)  
sys.setdefaultencoding('utf8')

DEFAULT_AN_LOCATION = 'Analysis'
NGRAMS = [1, 2, 3, 4]


def get_text_ngram(content, n=1):
    """
    Create and return sets of ngram words from the given field of a review
    """

    tokenizer = RegexpTokenizer(r'\w+')
    lemmatizer = WordNetLemmatizer()
    tokens = tokenizer.tokenize(content)
    lemma = [lemmatizer.lemmatize(t) for t in tokens]

    if n == 1:
        return lemma
    elif n in range(2, 10):
        return [' '.join(grams) for grams in ngrams(lemma, n)]
    else:
        print "The number of grams must be a integer between 1 and 10"


def plot_rating_distro(trainset):
    """
    Bar plot for the score distribution
    """

    print "Plot the distribution of ratings\n"

    ratings = np.array([r.rating for r in trainset])

    d = np.diff(np.unique(ratings)).min()
    left_of_first_bin = ratings.min() - float(d)/2
    right_of_last_bin = ratings.max() + float(d)/2

    plt.hist(ratings, np.arange(left_of_first_bin, right_of_last_bin + d, d), normed=True)

    plt.xlabel('Rating score over 5')
    plt.ylabel('Distribution Percentage')
    plt.title("Rating distribution")
    formatter = FuncFormatter(utils.to_percent)
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.tight_layout()
    plt.savefig(DEFAULT_AN_LOCATION + "/rating_distro.png", format='png', dpi=300)
    plt.close()

    # Plot correlation between num and average rating by product
    plt.plot(sorted(ratings), np.arange(0.0, 1.0, 1/float(len(ratings))))
    plt.xlabel('Rating score over 5')
    plt.ylabel('Cumulative distribution')
    plt.title("Rating cumulative distribution")
    plt.tight_layout()
    axes = plt.gca()
    axes.set_xlim([0, 6])
    axes.set_ylim([0, 1.2])
    plt.savefig(DEFAULT_AN_LOCATION + "/rating_cum.png", format='png', dpi=300)
    plt.close()


def plot_product_distro(trainset):
    """
    Bar plot for the product distribution and correlation with score
    """

    print "Plot the distribution of products"

    rating = []
    number = []
    index = []
    prod_len = dict()
    prod_sum = dict()

    for r in trainset:
        if r.product not in prod_sum:
            prod_sum[r.product] = r.rating
            prod_len[r.product] = 1
        else:
            prod_sum[r.product] += r.rating
            prod_len[r.product] += 1

    i = 0
    for key in prod_sum:
        rating.append(prod_sum[key] / float(prod_len[key]))
        number.append(prod_len[key])
        index.append(i)
        i += 1

    # Normalize the distributions
    rating = np.array(rating)
    number = np.array(number)
    rating_normed = rating / np.linalg.norm(rating)
    number_normed = number / np.linalg.norm(number)

    # Plot correlation between num and average rating by product
    print "Pearson correlation factor between number of reviews and average score by product is: " + str(
        abs(np.corrcoef(rating_normed, number_normed)[0, 1])) + "\n"
    plt.plot(number, rating, ".")
    plt.xlabel('Number of reviews per product')
    plt.ylabel('Average score per product')
    plt.tight_layout()
    plt.savefig(DEFAULT_AN_LOCATION + "/product_corr.png", format='png', dpi=300)
    plt.close()

    # Plot distribution of reviews number by product
    d = 0.5
    left_of_first_bin = rating.min() - float(d) / 2
    right_of_last_bin = rating.max() + float(d) / 2
    weights = np.ones_like(rating) / float(number.size)
    plt.hist(rating, np.arange(left_of_first_bin, right_of_last_bin + d, d), weights=weights)
    plt.xlabel('Average score per product')
    plt.ylabel('Distribution percentage')
    axes = plt.gca()
    axes.set_xlim([0, 6])
    formatter = FuncFormatter(utils.to_percent)
    axes.yaxis.set_major_formatter(formatter)
    plt.tight_layout()
    plt.savefig(DEFAULT_AN_LOCATION + "/product_rating.png", format='png', dpi=300)
    plt.close()

    # Plot distribution of average rating by product
    d = 1
    left_of_first_bin = number.min() - float(d) / 2
    right_of_last_bin = number.max() + float(d) / 2
    weights = np.ones_like(number) / float(number.size)
    plt.hist(number, np.arange(left_of_first_bin, right_of_last_bin + d, d), weights=weights)
    plt.xlabel('Number of reviews per product')
    plt.ylabel('Distribution percentage')
    axes = plt.gca()
    axes.set_xlim([0, 200])
    formatter = FuncFormatter(utils.to_percent)
    axes.yaxis.set_major_formatter(formatter)
    plt.tight_layout()
    plt.savefig(DEFAULT_AN_LOCATION + "/product_num.png", format='png', dpi=300)
    plt.close()

    # Plot Cumulative graph
    rating = np.array(rating)
    rating, index = (list(t) for t in zip(*sorted(zip(rating, index))))

    plt.plot(rating, ".")
    plt.xlabel('Product number')
    plt.ylabel('Average score')
    plt.tight_layout()
    axes = plt.gca()
    axes.set_ylim([0, 6])
    plt.savefig(DEFAULT_AN_LOCATION + "/product_cum.png", format='png', dpi=300)
    plt.close()


def plot_author_distro(trainset):
    """
    Bar plot for the author distribution and correlation with score
    """

    print "Plot the distribution of authors"

    author_sum = dict()
    author_len = dict()
    rating = []
    number = []
    index = []

    for r in trainset:
        if r.author not in author_sum:
            author_sum[r.author] = r.rating
            author_len[r.author] = 1
        else:
            author_sum[r.author] += r.rating
            author_len[r.author] += 1

    i = 0
    for key in author_sum:
        rating.append(author_sum[key] / float(author_len[key]))
        number.append(author_len[key])
        index.append(i)
        i += 1

    # Normalize the distributions
    rating = np.array(rating)
    number = np.array(number)
    rating_normed = rating / np.linalg.norm(rating)
    number_normed = number / np.linalg.norm(number)
    # rating, indexes2 = (list(t) for t in zip(*sorted(zip(rating, indexes))))

    # Plot correlation between num and average rating by product
    print "Pearson correlation factor between number of reviews and average score by author is: "\
          + str(abs(np.corrcoef(rating_normed, number_normed)[0, 1])) + "\n"
    plt.plot(number, rating, ".")
    plt.xlabel('Number of reviews per author')
    plt.ylabel('Average score per author')
    plt.tight_layout()
    plt.savefig(DEFAULT_AN_LOCATION + "/author_corr.png", format='png', dpi=300)
    plt.close()

    # Plot distribution of average rating by product
    d = 0.5
    left_of_first_bin = rating.min() - float(d)/2
    right_of_last_bin = rating.max() + float(d)/2
    weights = np.ones_like(rating)/float(number.size)
    plt.hist(rating, np.arange(left_of_first_bin, right_of_last_bin + d, d), weights=weights)
    plt.xlabel('Average score per author')
    plt.ylabel('Distribution Percentage')
    axes = plt.gca()
    axes.set_xlim([0, 6])
    formatter = FuncFormatter(utils.to_percent)
    axes.yaxis.set_major_formatter(formatter)
    plt.tight_layout()
    plt.savefig(DEFAULT_AN_LOCATION + "/author_rating.png", format='png', dpi=300)
    plt.close()

    # Plot distribution of reviews number by product
    d = 1
    left_of_first_bin = number.min() - float(d)/2
    right_of_last_bin = number.max() + float(d)/2
    weights = np.ones_like(number)/float(number.size)
    plt.hist(number, np.arange(left_of_first_bin, right_of_last_bin + d, d), weights=weights)
    plt.xlabel('Number of reviews per author')
    plt.ylabel('Distribution Percentage')
    axes = plt.gca()
    axes.set_xlim([0, 50])
    formatter = FuncFormatter(utils.to_percent)
    axes.yaxis.set_major_formatter(formatter)
    plt.tight_layout()
    plt.savefig(DEFAULT_AN_LOCATION + "/author_num.png", format='png', dpi=300)
    plt.close()

    # Plot cumulative distro
    rating = np.array(rating)
    rating, index = (list(t) for t in zip(*sorted(zip(rating, index))))
    plt.plot(rating, ".")
    plt.xlabel('Author ID')
    plt.ylabel('Average score')
    plt.tight_layout()
    axes = plt.gca()
    axes.set_ylim([0, 6])
    plt.savefig(DEFAULT_AN_LOCATION + "/author_cum.png", format='png', dpi=300)
    plt.close()


def plot_date_distro(trainset):
    """
    Bar plot for the date distribution and correlation with score
    """

    print "Plot the distribution of date\n"

    # Re-arrange the data
    date_sum = dict()
    date_len = dict()
    date = []
    for r in trainset:
        date_tab = [int(e) for e in r.date.translate(None, ';,').split()]
        date_num = datetime.date(date_tab[2], date_tab[0], date_tab[1])
        date.append(date_num)
        if not date_num.toordinal() in date_sum:
            date_sum[date_num.toordinal()] = r.rating
            date_len[date_num.toordinal()] = 1
        else:
            date_sum[date_num.toordinal()] += r.rating
            date_len[date_num.toordinal()] += 1

    rating_av = []
    index = []
    date = np.array(date)
    for key in date_sum:
        rating_av.append(date_sum[key] / float(date_len[key]))
        index.append(key)

    index, rating_av = (list(t) for t in zip(*sorted(zip(index, rating_av))))
    date_av = np.array([datetime.date.fromordinal(o) for o in index])

    # Plot average rating evolution over time
    fig, ax = plt.subplots()
    ax.plot(date_av, rating_av, ".")
    lp_window = 50
    ax.plot(date_av[(lp_window/2):-(lp_window/2)+1], utils.lp_filter(rating_av, lp_window))
    plt.xlabel('Date')
    plt.ylabel('Average score')
    plt.tight_layout()
    ax.fmt_xdata = DateFormatter('%Y-%m')
    ax.set_ylim([0, 6])
    plt.savefig(DEFAULT_AN_LOCATION + "/date.png", format='png', dpi=300)
    plt.close()

    # Plot date cumulative distribution
    fig, ax = plt.subplots()
    ax.plot(sorted(date), np.arange(0.0, 1.0, 1/float(len(date))))
    plt.xlabel('Date')
    plt.ylabel('Cumulative distribution')
    formatter = FuncFormatter(utils.to_percent)
    ax.yaxis.set_major_formatter(formatter)
    ax.fmt_xdata = DateFormatter('%Y-%m')
    plt.tight_layout()
    plt.savefig(DEFAULT_AN_LOCATION + "/date_cum.png", format='png', dpi=300)
    plt.close()

    # Plot date distribution
    d = 100
    left_of_first_bin = np.array(index).min() - float(d) / 2
    right_of_last_bin = np.array(index).max() + float(d) / 2
    fig, ax = plt.subplots()
    ax.hist(date, np.arange(left_of_first_bin, right_of_last_bin + d, d), normed=True)
    plt.xlabel('Date')
    plt.ylabel('Distribution Percentage')
    formatter = FuncFormatter(utils.to_percent)
    ax.yaxis.set_major_formatter(formatter)
    ax.fmt_xdata = DateFormatter('%Y-%m')
    plt.tight_layout()
    plt.savefig(DEFAULT_AN_LOCATION + "/date_distro.png", format='png', dpi=300)
    plt.close()


def detect_language(content):
    """
    Select and return the language of a string with the highest probability using nltk
    """

    l_ratios = {}
    tokens = wordpunct_tokenize(content)
    words = [word.lower() for word in tokens]

    for l in stopwords.fileids():
        stopwords_set = set(stopwords.words(l))
        words_set = set(words)
        common_set = words_set.intersection(stopwords_set)

        l_ratios[l] = len(common_set)

    return max(l_ratios, key=l_ratios.get)


def plot_languages(trainset):
    """
    Bar plot for languages in the review content
    """

    print "Plot the distribution of language in the review content"

    language = []
    i = 0

    for r in trainset:

        language.append(detect_language(r.content))

        if i % 50 == 0:
            print "Finding language for review " + str(i) + "/" + str(len(trainset))
        i += 1

    language_count = Counter(language)

    labels, values = zip(*language_count.most_common(5))
    values, labels = (list(t) for t in zip(*sorted(zip(values, labels), reverse=True)))

    indexes = np.arange(len(labels))
    width = 1

    bar_list = plt.bar(indexes, values, width)
    cols = plt.rcParams['axes.prop_cycle']
    col_list = []
    for v in cols:
        col_list.append(v)
    for i, b in enumerate(bar_list):
        b.set_color(col_list[i % len(col_list)]["color"])
    plt.xticks(indexes + width * 0.5, labels)
    plt.xlabel('Language')
    plt.ylabel('Number of reviews')
    plt.tight_layout()
    plt.savefig(DEFAULT_AN_LOCATION + "/languages.png", format='png', dpi=300)
    plt.close()
    print ""

    with open(os.path.join(DEFAULT_AN_LOCATION, "language.txt"), "w") as f:
        for k, v in language_count.most_common():
            f.write("{} {}\n".format(k, v))


def plot_content_len_distro(trainset):
    """
    Bar plot for the content length distribution and correlation with score
    """

    print "Plot the distribution of content length"
    length = []
    length_by_rating = [[], [], [], [], []]
    rating = [r.rating for r in trainset]

    for r in trainset:
        l = len(r.content.split())
        length.append(l)
        length_by_rating[r.rating - 1].append(l)

    r = [1, 2, 3, 4, 5]
    len_av = [np.mean(l) for l in length_by_rating]
    len_err = [np.std(l) for l in length_by_rating]
    length, rating = (list(t) for t in zip(*sorted(zip(length, rating), reverse=True)))

    # Plot correlation between num and average rating by product
    print "Pearson correlation factor between length of a review and average score is: " + \
          str(abs(np.corrcoef(rating, length)[0, 1])) + "\n"
    plt.errorbar(r, len_av, len_err, ecolor=utils.get_style_colors()[1])
    plt.ylabel('Average length of a review')
    plt.title("Evolution of review length with rating")
    plt.xlabel('Rating')
    plt.tight_layout()
    axes = plt.gca()
    axes.set_xlim([0, 6])
    plt.savefig(DEFAULT_AN_LOCATION + "/len_corr.png", format='png', dpi=300)
    plt.close()

    plt.plot(sorted(length), np.arange(0.0, 1.0, 1/float(len(length))))
    plt.title("Review length cumulative distribution")
    plt.xlabel('Number of words in review')
    plt.ylabel('Cumulative distribution')
    plt.tight_layout()
    axes = plt.gca()
    axes.set_xlim([min(length)-10, max(length)+10])
    axes.set_ylim([0, 1.2])
    plt.savefig(DEFAULT_AN_LOCATION + "/len_cum.png", format='png', dpi=300)
    plt.close()


def plot_content_tfidf(trainset):

    reviews = [r.content for r in trainset]
    ratings = [r.rating - 3 for r in trainset]
    max_print = 5
    max_plot = 1000
    term_occ_norm_sw = []
    term_occ_norm = []

    for n in NGRAMS:
        for j in range(2):

            if j == 0:
                print "Compute TFIDF for " + str(n) + "-gram words and stop words"
                stop_words = text.ENGLISH_STOP_WORDS.union(["s", "t", "2", "m", "ve"])
                tfidf_vec = TfidfVectorizer(tokenizer=get_text_ngram, ngram_range=(n, n), stop_words=stop_words)
            else:
                print "Compute TFIDF for " + str(n) + "-gram words without stop words"
                tfidf_vec = TfidfVectorizer(tokenizer=get_text_ngram, ngram_range=(n, n))
            tfs = tfidf_vec.fit_transform(reviews)

            names = tfidf_vec.get_feature_names()
            idf = tfidf_vec.idf_
            tfidf = np.array(tfs.sum(axis=0).transpose() / len(names)).flatten().tolist()
            n_f = len(reviews)
            df = ((n_f + 1) / (np.e ** (idf - 1))) - 1
            df_ratio = df / n_f
            if j == 0:
                term_occ_norm_sw.append(sorted(df_ratio, reverse=True)[0:max_plot])
            else:
                term_occ_norm.append(sorted(df_ratio, reverse=True)[0:max_plot])
            corr = np.array(np.matrix(ratings) * tfs).transpose().flatten().tolist()

            # Sort and print ngrams by TFIDF average
            print "Sort and print results by TFIDF value for " + str(n) + "-gram words"
            tfidf_s, names_s, idf_s = (list(t) for t in zip(*sorted(zip(tfidf, names, idf), reverse=True)))

            if j == 0:
                tfidf_file = open(os.path.join(DEFAULT_AN_LOCATION, str(n) + "-gram_tfidf_sw.txt"), "w")
            else:
                tfidf_file = open(os.path.join(DEFAULT_AN_LOCATION, str(n) + "-gram_tfidf.txt"), "w")
            st = "\n --- Highest TFIDF for " + str(n) + "-grams ---\n\n"
            st += "| TFIDF\t\t\t| TF\t\t\t| IDF\t\t| NGRAM\n"
            st += "-------------------------------------------\n"
            for i in range(max_print):
                st += "| " + utils.n2str(tfidf_s[i]) + "\t\t| " + utils.n2str(tfidf_s[i]/idf_s[i]) + "\t\t| " +\
                      utils.n2str(idf_s[i]) + "\t\t| " + str(names_s[i]) + "\n"
            tfidf_file.write(st)
            tfidf_file.close()

            # Sort and print correlation between ngrams and rating
            print "Sort and print correlation between rating and " + str(n) + "-gram words"
            corr, names = (list(t) for t in zip(*sorted(zip(corr, names), reverse=True)))

            if j == 0:
                corr_file = open(os.path.join(DEFAULT_AN_LOCATION, str(n) + "-gram_corr_sw.txt"), "w")
            else:
                corr_file = open(os.path.join(DEFAULT_AN_LOCATION, str(n) + "-gram_corr.txt"), "w")
            st = "\n --- Highest " + str(n) + "-grams * rating products ---\n\n"
            st += "|  TFIDF * Rating\t|  NGRAM\n"
            st += "----------------------------\n"
            for i in range(int(max_print / 2)):
                st += "| " + utils.n2str(corr[i]) + "\t\t\t| " + str(names[i]) + "\n"
            st += "\n --- Lowest " + str(n) + "-grams * rating products ---\n\n"
            st += "|  TFIDF * Rating\t|  NGRAM\n"
            st += "----------------------------\n"
            for i in range(int(max_print / 2)):
                st += "| " + utils.n2str(corr[-i-1]) + "\t\t\t| " + str(names[-i-1]) + "\n"
            corr_file.write(st)
            corr_file.close()

    # Plot tfidf histogram
    print "Plot term distribution"
    lw = 2 * int(plt.rcParams["lines.linewidth"])
    x = range(lw, max_plot + lw)
    for i in xrange(len(term_occ_norm)):
        plt.plot(x, term_occ_norm[i], color=utils.get_style_colors()[i], label='%s-gram' % str(i+1))
    plt.legend()
    plt.xlabel('Term Index')
    plt.ylabel('Percentage of Occurrences')
    axes = plt.gca()
    axes.set_xlim([0, 1000])
    formatter = FuncFormatter(utils.to_percent)
    axes.yaxis.set_major_formatter(formatter)
    plt.tight_layout()
    plt.savefig(DEFAULT_AN_LOCATION + "/ngram_distro.png", format='png', dpi=300)
    plt.close()

    for i in xrange(len(term_occ_norm)):
        plt.plot(x, term_occ_norm_sw[i], color=utils.get_style_colors()[i], label='%s-gram' % str(i+1))
    plt.legend()
    plt.xlabel('Term Index')
    plt.ylabel('Percentage of Occurrences')
    axes = plt.gca()
    axes.set_xlim([0, 1000])
    formatter = FuncFormatter(utils.to_percent)
    axes.yaxis.set_major_formatter(formatter)
    plt.tight_layout()
    plt.savefig(DEFAULT_AN_LOCATION + "/ngram_distro_sw.png", format='png', dpi=300)
    plt.close()


def main():
    """
    Load data and create features
    """

    dataset = data.load_pickled_data()
    train_set = dataset['train']
    utils.make_dir(DEFAULT_AN_LOCATION)

    plot_rating_distro(train_set)
    plot_product_distro(train_set)
    plot_author_distro(train_set)
    plot_date_distro(train_set)
    plot_languages(train_set)
    plot_content_len_distro(train_set)
    plot_content_tfidf(train_set)

if __name__ == '__main__':
    main()
