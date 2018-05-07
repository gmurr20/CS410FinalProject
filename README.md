# CS410FinalProject
## Summary
For CS 410, we created a script to analyze the essays of teachers looking for the approval and funding of DonorsChoose.org.  This script can make a prediction on whether their project was approved by using various NLP techniques.  For the scope of this class, we fixed our classifier to a Multi-layer Perceptron (changed from an SVM because as the feature dimensionality increased, the training took too long and the pkl file for each classifier was too large for github).  We used NLP techniques for part of speech tags, term frequencies, n-gram analysis, and LDA topic analysis and used the results for each as features in our classifier and then outputted various metrics which were learned in this class.  These metrics include accuracy, precision, recall, and f1 score.
## Overview of functions and Implementation
An overview of the functions can be found in the comments of the code.  Above each function, we describe how the function is used and describe its parameters and return values.  Here are the major functions documented here for our graders convenience:
* stemEssays(essays)

Stems the words in each essay.  This is only run once and the returned object is saved as a pkl file.  This function takes around an hour to run, as stemming each word in each essay for our data is very time consuming.  We used nltk.stem to help us stem each word.

@:param essays - dictionary of essays with the format {project id: [essay 1, 2, 3, 4]}

@:return a dictionary with the format {project id: [ list of stemmed words for essay 1, list of stemmed wordsfor essay 2, 3, 4 ] }
* wordCounts(essays, stopSet, x_train, y_train, x_test)

Counting the word frequencies in the approved essays and non-approved essays and normalizing it by the length of the corpus, we take the most common 300 words in each category (ignoring stopwords) and take any words that are not intersecting for approved and non-approved.  For example, "student" is a very common word in both approved and non-approved essays, so that word is not used for our features.  We then take our feature vocabulary and transform the training and test data to word counts for an MLP to classify.

@:param essays- a dictionary with the format {project id: [ [list of stemmed words]1, [list of stemmed words]2, []3, []4 ]

@:param stopSet- a set of stopwords

@:param x_train- a list of project ids for training data

@:param y_train- a list of 1s and 0s representing whether project was approved

@:param x_test- a list of project ids for test data

@:returns an array of 30 features which represent various word counts

## Usage Documentation
Due to how large the original training files were, we had to trim our training data. Stemming the essays takes around an hour to run, so we recommend downloading the pkl file and using the already stemmed object instead.  Here are the steps to get our script running on your machine:
1. Download the **essayPrediction.py** file, **obj/stemmedEssays.pkl** file, the **trimmedTrain.csv** file, and the **resources.csv.zip** file from our github repository. Make sure that each file is in the same directory, and that the **stemmedEssays.pkl** file is in its own obj directory.
1. Unzip the resources.csv.zip file.
1. Make sure you have the following 3rd party libaries installed for python, otherwise the script will fail.
    1. sklearn http://scikit-learn.org/stable/install.html
    1. nltk https://www.nltk.org/install.html
    1. pickle https://docs.python.org/3/library/pickle.html
    1. numpy
1. In your terminal, run **python essayPrediction.py *-argument* *--pred***
    1. *--pos* will use the part of speech tagging classifier
    1. *--lda int* will use the lda topic classifier
    1. *--ngram int* will use the bi-gram analysis classifier
    1. *--tf* will use the term frequency classifier
1. The following output, should give you extensive stats on how the classifier performed on the training and test data. If *--pred* is specified, a csv file of predictions will be made. In that file you can view your predictions.
