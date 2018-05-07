import argparse
import csv
import string
import pickle
import os
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import random

'''
Initialize for passing in arguments
'''
def initArgs():
	parser = argparse.ArgumentParser(description='Predict whether teacher essays are approved in the trimmedTrain.csv file using an NLP technique')
	parser.add_argument('--pos', action="store_true", default=False, help="use this argument to use term frequency with part of speech tagging")
	parser.add_argument('--lda', action="store_true", default=False, help="use this argument with an integer k to use LDA topic analysis")
	parser.add_argument('--ngram', type=int, help="use this argument with an integer n to use n-gram analysis")
	parser.add_argument('--tf', action="store_true", default=False, help="use this argument to use term frequencies")
	parser.add_argument('--pred', action="store_true", help="use this argument if you want a predictions file")
	args = parser.parse_args()
	return args

'''
Save object as a pkl file
#https://stackoverflow.com/questions/19201290/how-to-save-a-dictionary-to-a-file
'''
def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

'''
Load object as a pkl file
#https://stackoverflow.com/questions/19201290/how-to-save-a-dictionary-to-a-file
'''
def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

'''
Loads the resources.csv file that contains the resources requested for each project
returns a dictionary with the following format {project id : [description, quantity, price]}
'''
def loadResources():
	resources = {}
	with open('resources.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile)
		skipFirstRow = True
		for row in reader:
			if skipFirstRow:
				skipFirstRow = False
				continue
			projectId = row[0]
			if projectId not in resources:
				resources[projectId] = [row[1:]]
			else:
				resources[projectId] = resources[projectId]+[row[1:]]		
	return resources

'''
Loads the essays for each project
returns a dictionary  with the following format {project id: [essay 1, 2, 3, 4]}
'''
def loadEssays():
	data = {}
	rows = []
	with open('trimmedTrain.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile)
		skipFirstRow = True
		badCount = 0
		goodCount = 0
		for row in reader:
			if skipFirstRow:
				skipFirstRow = False
				continue
			projectId = row[0]
			essays = [str(r) for r in row[9:13]]
			data[projectId] = essays
	return data

'''
Stems the essays: Only run once to stem essays and saves obj as a pkl
@:param essays - dictionary of essays with the format {project id: [essay 1, 2, 3, 4]}
@:return a dictionary with the format {project id: [ list of stemmed words for essay 1, list of stemmed words for essay 2, list 3, list 4 ] }
'''
def stemEssays(essays):
	stemmedEssays = {}
	
	#see if files have already been stemmed
	if os.path.isfile('obj/stemmedEssays.pkl'):
		print("Loading stemmed essays object (takes about a minute)")
		stemmedEssays = load_obj('stemmedEssays')
		print("Loading done\n")
		return stemmedEssays

	print("-------------stemming essays-------------")
	i = 0.0
	for key, value in essays.items():
		essayList = value
		porter = PorterStemmer()
		documents = []
		for essay in essayList:
			stemmedEssay = essay.lower()
			#strip punctuation
			stemmedEssay = stemmedEssay.translate(None, string.punctuation)
			#stems each word into an array
			document = [porter.stem(word) for word in stemmedEssay.decode('utf-8').split()]
			documents.append(document)
		stemmedEssays[key] = documents
		if (i/len(essays.items())*100) % 5 == 0:
			print(str(i/len(essays.items())*100)+'% done')
		i+=1
	print("--------stemming essays completed--------")
	save_obj(stemmedEssays, 'stemmedEssays')
	return stemmedEssays

'''
Does a 70/30 train test split on the project ids
@:returns X_train, X_test, y_train, y_test where X is a list of project ids and y is approved/not approved
'''
def trainTestSplit():
	ids = []
	yesNo = []
	with open('trimmedTrain.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile)
		skipFirstRow = True
		for row in reader:
			if skipFirstRow:
				skipFirstRow = False
				continue
			ids.append(row[0])
			yesNo.append(int(row[len(row)-1]))
	X_train, X_test, y_train, y_test = train_test_split(ids, yesNo, test_size=.3)
	return X_train, X_test, y_train, y_test

'''
Counting the word frequencies in the approved essays and non-approved essays and normalizing it by the length of the corpus, we take the top 1000
words from the approved category and top 1000 words from the non-approved category and use these as our feature vocabulary.
We then take our feature vocabulary and transform the training and test data to word counts normalized by essay for an SVM to classify.
@:param essays- a dictionary with the format {project id: [ [list of stemmed words]1, [list of stemmed words]2, []3, []4 ]
@:param stopSet- a set of stopwords
@:param x_train- a list of project ids for training data
@:param y_train- a list of 1s and 0s representing whether project was approved
@:param x_test- a list of project ids for test data
@:returns an array of 30 features which represent various word counts
'''
def wordCounts(essays, stopSet, x_train, y_train, x_test, resources):
	#create a dictionary with format {word: count in corpus}
	goodCount = {}
	badCount = {}
	i = 0
	for x_id in x_train:
		for essay in essays[x_id]:
			for word in essay:
				if word in stopSet or len(word) <= 2:
					continue
				if y_train[i] == 1:
					if word not in goodCount:
						goodCount[word] = 0
					goodCount[word] += 1
				else:
					if word not in badCount:
						badCount[word] = 0
					badCount[word] += 1
		i+=1

	for key, value in goodCount.items():
		goodCount[key] = 1.0*value/len(x_train)

	for key, value in badCount.items():
		badCount[key] = 1.0*value/len(x_train)

	#sort by count
	topGood = []
	for key, value in sorted(goodCount.items(), key=lambda (k,v): (-v,k)):
		topGood.append( key )
	topBad = []
	for key, value in sorted(badCount.items(), key=lambda (k,v): (-v,k)):
		topBad.append( key )

	gSet = set(topGood[:1000])
	bSet = set(topBad[:1000])

	finalWords = gSet.union(bSet)
	mapToIdx = {}
	i=0
	for w in finalWords:
		mapToIdx[w] = i
		i+=1
	
	#transform training data
	train_features = []
	for x_id in x_train:
		feat = np.zeros(len(finalWords)+1)
		myLen = 0
		for essay in essays[x_id]:
			myLen += len(essay)
			for word in essay:
				if word in finalWords:
					feat[mapToIdx[word]] += 1
		for f in feat:
			f = 1.0*f/myLen
		total = 0
		for r in resources[x_id]:
			total += float(r[1]) * float(r[2])
		feat[len(feat)-1] = total
		train_features.append(feat)

	#transform testing data
	test_features = []
	for x_id in x_test:
		feat = np.zeros(len(finalWords)+1)
		myLen = 0
		for essay in essays[x_id]:
			myLen += len(essay)
			for word in essay:
				if word in finalWords:
					feat[mapToIdx[word]] += 1
		total = 0
		for r in resources[x_id]:
			total += float(r[1]) * float(r[2])
		feat[len(feat)-1] = total
		for f in feat:
			f = 1.0*f/myLen
		test_features.append(feat)
	return train_features, test_features

#https://pythonprogramming.net/natural-language-toolkit-nltk-part-speech-tagging/
def partOfSpeechTags():
	return

#https://stackoverflow.com/questions/32441605/generating-ngrams-unigrams-bigrams-etc-from-a-large-corpus-of-txt-files-and-t
def ngramAnalysis():
	return

#mp3 or mp2
def ldaTopicAnalysis():
	return

def main():
	args = initArgs()
	stopSet = set(stopwords.words('english'))
	resources = loadResources()
	notStemmedEssays = loadEssays()
	X_train, X_test, y_train, y_test = trainTestSplit()
	essays = stemEssays(notStemmedEssays)
	
	fileName = 'predictions'
	trainPredictions = None
	testPredictions = None
	trainProbs = None
	testProbs = None
	clf = None

	#term frequency
	if args.tf:
		print "---------Term Frequency (takes about 2 minutes)---------\n"
		transformed_train, transformed_test = wordCounts(essays, stopSet, X_train, y_train, X_test, resources)
		clf = MLPClassifier(solver='lbfgs', activation='relu', alpha=.1, early_stopping=True)
		clf.fit(transformed_train, y_train)
		
		fileName += 'Tf'
		trainPredictions = clf.predict(transformed_train)
		testPredictions = clf.predict(transformed_test)
		trainProbs = clf.predict_proba(transformed_train)
		testProbs = clf.predict_proba(transformed_test)

		print "METRICS"
		print "Training accuracy: "+str(accuracy_score(y_train, trainPredictions))
		print "Test accuracy: "+str(accuracy_score(y_test, testPredictions))+'\n'
		print classification_report(y_test, testPredictions)
		print "----------------Term Frequency Completed----------------\n"
	#part of speech
	elif args.pos:
		print "---------Part of Speech (takes about 2 minutes)---------\n"
		transformed_train, transformed_test = partOfSpeechTags(essays, stopSet, X_train, y_train, X_test, resources)
		clf = MLPClassifier(solver='lbfgs', activation='relu', alpha=.1, early_stopping=True)
		clf.fit(transformed_train, y_train)

		fileName += 'Pos'
		trainPredictions = clf.predict(transformed_train)
		testPredictions = clf.predict(transformed_test)
		trainProbs = clf.predict_proba(transformed_train)
		testProbs = clf.predict_proba(transformed_test)

		print "METRICS"
		print "Training accuracy: "+str(accuracy_score(y_train, trainPredictions))
		print "Test accuracy: "+str(accuracy_score(y_test, testPredictions))+'\n'
		print classification_report(y_test, testPredictions)
		print "----------------Part of Speech Completed----------------\n"
	#lda topic	
	elif args.lda > 1:
		print "------------LDA Topic (takes about 2 minutes)-----------\n"
		transformed_train, transformed_test = ldaTopicAnalysis(essays, stopSet, X_train, y_train, X_test, resources)
		clf = MLPClassifier(solver='lbfgs', activation='relu', alpha=.1, early_stopping=True)
		clf.fit(transformed_train, y_train)

		fileName += 'Lda'
		trainPredictions = clf.predict(transformed_train)
		testPredictions = clf.predict(transformed_test)
		trainProbs = clf.predict_proba(transformed_train)
		testProbs = clf.predict_proba(transformed_test)

		print "METRICS"
		print "Training accuracy: "+str(accuracy_score(y_train, trainPredictions))
		print "Test accuracy: "+str(accuracy_score(y_test, testPredictions))+'\n'
		print classification_report(y_test, testPredictions)
		print "------------------LDA Topic Completed------------------\n"

	elif args.ngram > 1:
		print "----------Ngram Topic (takes about 2 minutes)----------\n"
		transformed_train, transformed_test = ngramAnalysis(essays, stopSet, X_train, y_train, X_test, resources)
		clf = MLPClassifier(solver='lbfgs', activation='relu', alpha=.1, early_stopping=True)
		clf.fit(transformed_train, y_train)

		fileName += 'Ngram'
		trainPredictions = clf.predict(transformed_train)
		testPredictions = clf.predict(transformed_test)
		trainProbs = clf.predict_proba(transformed_train)
		testProbs = clf.predict_proba(transformed_test)

		print "METRICS"
		print "Training accuracy: "+str(accuracy_score(y_train, trainPredictions))
		print "Test accuracy: "+str(accuracy_score(y_test, testPredictions))+'\n'
		print classification_report(y_test, testPredictions)
		print "-------------------Ngram Completed--------------------\n"

	else:
		print "You must specify an NLP method"
		return

	if args.pred != None:
		fileName += '.csv'
		with open(fileName, 'w+') as file:
			line = 'Training Data\nProject ID, Probability of 0, Probability of 1, Prediction, Actual\n'
			file.write(line)
			for i in range(len(X_train)):
				pId = X_train[i]
				prob0 = trainProbs[i][0]
				prob1 = trainProbs[i][1]
				pred = trainPredictions[i]
				actual = y_train[i]
				line = pId + ',' + str(prob0) + ',' + str(prob1) + ',' + str(pred) + ',' + str(actual) + '\n'
				file.write(line)

			line = '\nTest Data\n'
			file.write(line)
			for i in range(len(y_test)-1):
				pId = X_test[i]
				prob0 = testProbs[i][0]
				prob1 = testProbs[i][1]
				pred = testPredictions[i]
				actual = y_test[i]
				line = pId + ',' + str(prob0) + ',' + str(prob1) + ',' + str(pred) + ',' + str(actual) + '\n'
				file.write(line)

		print "Predictions file done"

	return





if __name__ == "__main__":
	#python essayPrediction.py -nlpMethod
	main()