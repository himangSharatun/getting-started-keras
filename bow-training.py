from sklearn.feature_extraction.text import CountVectorizer
from tokenizer import tokenizeSentence as tokenize
import csv
import json
import pandas
import numpy

sentences = []

# with open('youtube_comment.csv') as csvfile:
#     readCSV = csv.reader(csvfile, delimiter=',')
#     for row in readCSV:
#         sentences.append(tokenize(row[0]))

X_training = []
Y = []
dataframe = pandas.read_csv('data/training/training-data.csv', header=None)
for i in xrange(len(dataframe[0])):
    X_training.append(dataframe[0][i])
sentences = numpy.array(X_training)

# print sentences[0:5]
 
vectorizer = CountVectorizer()
vectorizer.fit_transform(sentences).todense() 

with open ('vocabulary.json', 'w') as vocabFile:
    json.dump(vectorizer.vocabulary_ , vocabFile)

print "vocabulary is saved"