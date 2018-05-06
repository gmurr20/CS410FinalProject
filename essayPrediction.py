import argparse
import csv
import string
import pickle
from nltk.stem import PorterStemmer


'''
Initialize for passing in arguments
'''
def initArgs():
	parser = argparse.ArgumentParser(description='Predict whether teacher essays are approved in the train.csv file using an NLP technique')
	parser.add_argument('-p', action="store_true", default=False, help="use this argument to use part of speech tagging")
	parser.add_argument('-e', action="store_true", default=False, help="use this argument to use entity recognition")
	parser.add_argument('-m', action="store_true", default=False, help="use this argument to use phrase mining")
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
Stems the essays
@:param essays - dictionary of essays with the format {project id: [essay 1, 2, 3, 4]}
@:return a dictionary with the format {project id: [ [list of stemmed words]1, [list of stemmed words]2, []3, []4 ] }
'''
def stemEssays(essays):
	stemmedEssays = {}
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

def main():
	initArgs()
	resources = loadResources()
	essays = loadEssays()
	essays = stemEssays(essays)

if __name__ == "__main__":
	#python essayPrediction.py -nlpMethod
	main()