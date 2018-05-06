import argparse
import csv
import string
import pickle
import os
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np

'''
Initialize for passing in arguments
'''
def initArgs():
	parser = argparse.ArgumentParser(description='Predict whether teacher essays are approved in the train.csv file using an NLP technique')
	parser.add_argument('-p', action="store_true", default=False, help="use this argument to use part of speech tagging")
	parser.add_argument('-l', action="store_true", default=False, help="use this argument to use LDA topic")
	parser.add_argument('-n', type=int, help="use this argument to use n-gram analysis")
	parser.add_argument('-c', action="store_true", default=False, help="use this argument to use vocabulary word counts")
	args = parser.parse_args()
	return args

#https://stackoverflow.com/questions/19201290/how-to-save-a-dictionary-to-a-file
def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

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
	with open('train.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile)
		skipFirstRow = True
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
@:return a dictionary with the format {project id: [ [list of stemmed words]1, [list of stemmed words]2, []3, []4 ] }
'''
def stemEssays(essays):
	stemmedEssays = {}
	
	#see if files have already been stemmed
	if os.path.isfile('obj/stemmedEssays.pkl'):
		print("Loading stemmed essays object (takes about 6 minutes)")
		stemmedEssays = load_obj('stemmedEssays')
		print("Loading completed")
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
	projectIdList = []
	approved = []
	with open('train.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile)
		skipFirstRow = True
		for row in reader:
			if skipFirstRow:
				skipFirstRow = False
				continue
			projectIdList.append(row[0])
			approved.append(int(row[15]))
	X_train, X_test, y_train, y_test = train_test_split(projectIdList, approved, test_size=.3, random_state=0)
	return X_train, X_test, y_train, y_test

'''
Counting the word frequencies in the approved essays and non-approved essays and normalizing it by the length of the corpus, we take the most common
300 words in each category (ignoring stopwords) and take any words that are not intersecting for approved and non-approved.  For example, "student" is
a very common word in both approved and non-approved essays, so that word is not used for our features.  We then take our feature vocabulary and transform
the training and test data to word counts for an svm to classify
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

	#only take the words that are not overlapping
	gSet = set(topGood[:250])
	goodSet = set(topGood[:250])
	bSet = set(topBad[:250])
	intersection = set([])
	for s in gSet:
		if s in bSet:
			intersection.add(s)
			goodSet.remove(s)
			bSet.remove(s)

	finalWords = goodSet.union(bSet)
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
		total = resources[x_id][1] * resources[x_id][1] 
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
		total = resources[x_id][1] * resources[x_id][1] 
		feat[len(feat)-1] = total
		for f in feat:
			f = 1.0*f/myLen
		test_features.append(feat)
	return train_features, test_features

#https://pythonprogramming.net/natural-language-toolkit-nltk-part-speech-tagging/
def partOfSpeechTags():
	return

def ngramAnalysis():
	return

def ldaTopicAnalysis():
	return

def main():
	args = initArgs()
	stopSet = set(stopwords.words('english'))
	print(stopSet)
	resources = loadResources()
	essays = loadEssays()
	X_train, X_test, y_train, y_test = trainTestSplit()
	essays = stemEssays(essays)
	transformed_train, transformed_test = wordCounts(essays, stopSet, X_train, y_train, X_test, resources)
	print("training mlp")
	clf = clf = MLPClassifier(solver='lbfgs', alpha=.01, learning_rate='adaptive', early_stopping=False)
	clf.fit(transformed_train, y_train)
	print("training done")
	print(accuracy_score(y_train, clf.predict(transformed_train)))
	print(accuracy_score(y_test, clf.predict(transformed_test)))

if __name__ == "__main__":
	#python essayPrediction.py -nlpMethod
	main()