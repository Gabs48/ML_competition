"""Simple prediction script"""

__author__ = "Gabriel Urbain"
__copyright__ = "Copyright 2017, Gabriel Urbain"

__license__ = "MIT"
__version__ = "0.2"
__maintainer__ = "Gabriel Urbain"
__email__ = "gabriel.urbain@ugent.be"
__status__ = "Research"
__date__ = "September 1st, 2017"


import nonlinear
import preprocessing
import utils

import sys


def main(ft_filename=None):

	print "1. Load or create the features\n"
	if ft_filename is None:
		ft_filename = preprocessing.DEFAULT_FT_LOCATION + "/ft" + utils.timestamp() + ".pkl"
		test_ft, train_ft, target = preprocessing.create_ft()
		preprocessing.save_ft(test_ft, train_ft, target, filename=ft_filename)

	print "2. Train the model and save predictions"
	nonlinear.test(predict=True, ft_name=ft_filename)

if __name__ == '__main__':

	args = sys.argv

	if args > 1:
		main(args[1])
	else:
		main()
