# Machine Learning Kaggle Competition
Code of a Kaggle competition for the Machine Learning class:


## Python requirements:
- SciPy
- SciKit Learn
- NLTK

## Main script usage

### Create data File

~~~bash
python create_data_pickle.py
~~~

### Exploratory data analysis

~~~bash
python analysis.py
~~~

### Parameters search for feature extraction

For grid-search:
~~~bash
python feature_selection.py ct|ct_pd_au|reduce
~~~
To plot the grid-search results:
~~~bash
python feature_selection.py plot_ct|plot_ct_pd_au|plot_reduce \
    <grid_search_results_filename.pkl>
~~~

### Creation of a features file

~~~bash
python preprocessing.py ft
~~~

### Parameters search for linear models

For grid-search:
~~~bash
python linear.py gs lr|lr_all|lr_all_svd|lr_mixed_svd|rf_all
~~~
To plot the grid-search results:
~~~bash
python linear.py plot <grid_search_results_filename.pkl>
~~~
For a single test (edit file accordingly):
~~~bash
python linear.py test
~~~

### Parameters search for non-linear models

For grid-search:
~~~bash
python linear.py gs
~~~
To plot the grid-search results:
~~~bash
python linear.py plot <grid_search_results_filename.pkl>
~~~
For a single test (edit file accordingly):
~~~bash
python linear.py test
~~~
For an test with ensemble voting between different classifiers:
~~~bash
python linear.py vote
~~~

### Prediction

To create a prediction, edit the file to select the desired model and run:
~~~bash
python predict.py [features_filename.pkl]
~~~

All other documentation can be found in the scientific project report or in the
code directly
