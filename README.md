# CS410FinalProject
##Summary
For CS 410, we created a script to analyze the essays of teachers looking for approval and funding by DonorsChoose.org.  This script can make a prediction on whether their project was approved by using various NLP techniques.  For the scope of this class, we fixed our classifier to a Multi-layer Perceptron (changed from an SVM because as the feature dimensionality increased, the training took too long).  We used NLP techniques for part of speech tags, term frequencies, n-gram analysis, and LDA topic analysis and used the results for each as features in our classifier and then compared the results.
## Overview of functions and Implementation
An overview of the functions can be found in the comments of the code.  Above each function, we describe how the function is used and describe its parameters and return values.  Here are the major functions documented here for our graders convenience:
* stemEssays(essays)

Stems the words in each essay.  This is only run once and the returned object is saved as a pkl file.  This function takes around 12 hours to run, as stemming each word in each essay for our data is very time consuming.  We used nltk.stem to help us stem each word.

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
Due to how large the training files were for the essays, we were not able to upload it to the github repository.  If the instructor wishes to run our preprocess.py file, you can download the training data at https://www.kaggle.com/c/donorschoose-application-screening/data.  From there you can run the preprocess file as a normal python executable as long as the train.csv file is in the same directory.  Stemming the essays takes around 12 hours, and training the various classifiers can take up to 12 minutes. For this reason, we decided to upload pkl files that represent the different classifiers trained under the various NLP techniques so you can run and test our code in a timely manner.  Instructions to download our script and get it running without waiting 12 hours for the stemming of essays and preprocessing of features is as follows:
1. Download the 4 classifier pkl files from our github repository along with the essayPrediction.py file.
1. Download the train.csv file from https://www.kaggle.com/c/donorschoose-application-screening/data and place in the same directory.
1. In your terminal, run **python essayPrediction -*insert nlp technique here* -*project id***
    1. -p will use the part of speech tagging classifier
    1. -l will use the lda topic classifier
    1. -n will use the bi-gram analysis classifier
    1. -t will use the term frequency classifier
1. The following output, should give you the prediction for the following project id along with the actual prediction.  If no project id is given, it will return the accuracy over the training and test data.
