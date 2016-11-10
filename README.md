# Machine Learning Kaggle Competition
Code of the Kaggle competition for the Machine Learning course

## Principle

Used techniques are:
- N-grams
- vectorized dictionnary
- Logistical regression
Used libraries are:
- sklearn

## Usage

- Get the code:

~~~bash
git clone http://github.com/Gabs48/ML_competition
~~~

- Place the Data folder in the repository

- Create a pickle file from data:

~~~bash
python2 create_data_pickle.py
~~~

- Create a vectorized dictionnary (sklearn) of features (lines: reviews; columns: features). This step produces a feature file and a model file in the folder *Features*:

~~~bash
python2 features.py
~~~

- Create a regression model using the features and target review ratings. This step produces a regression model file and a score file for the training set in the folder *Models*:

~~~bash
python2 train.py <path_to_feature_file.pkl>
~~~

- Estimate the score of the two previous tests on a validation set. This step produces a validation score file for the validation set in the folder *Models*:

~~~bash
python2 validate.py <path_to_feature_model.pkl> <path_to_regression_model.pkl>
~~~

- Predict review rating for the test set. This step produces a prediction numpy matrix for the test set in the folder *Predictions*:

~~~bash
python2 predict.py <path_to_feature_model.pkl> <path_to_regression_model.pkl>
~~~

- Transform prediction matrix in CSV file for Kaggle competition:

~~~bash
python2 create_submission.py <path_to_predicted_results.npy> [<path_to_CSV_file.csv>]
~~~

## Documentation
A full technical explanation can be found in *Doc/report.pdf*. This document is however not updated for every commit of code.
