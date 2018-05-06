# CS410FinalProject
## Overview of functions
An overview of the functions can be found in the comments of the code.  Above each function, we describe how the function is used and describe its parameters and return values.  Here are the major functions documented here for our graders convenience:
* stemEssays(essays)

Stems the words in each essay.  This is only run once and the returned object is saved as a pkl file.  This function takes around 12 hours to run, as stemming each word in each essay for our data is very time consuming.  We used nltk.stem to help us stem each word.

@:param essays - dictionary of essays with the format {project id: [essay 1, 2, 3, 4]}

@:return a dictionary with the format {project id: [ list of stemmed words for essay 1, list of stemmed wordsfor essay 2, 3, 4 ] }
* wordCounts(essays, stopSet, x_train, y_train, x_test)

Counting the word frequencies in the approved essays and non-approved essays and normalizing it by the length of the corpus, we take the most common 300 words in each category (ignoring stopwords) and take any words that are not intersecting for approved and non-approved.  For example, "student" is a very common word in both approved and non-approved essays, so that word is not used for our features.  We then take our feature vocabulary and transform the training and test data to word counts for an svm to classify.

@:param essays- a dictionary with the format {project id: [ [list of stemmed words]1, [list of stemmed words]2, []3, []4 ]

@:param stopSet- a set of stopwords

@:param x_train- a list of project ids for training data

@:param y_train- a list of 1s and 0s representing whether project was approved

@:param x_test- a list of project ids for test data

@:returns an array of 30 features which represent various word counts
