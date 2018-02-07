from sklearn.feature_extraction.text import CountVectorizer
import json

vocab_path = 'vocabulary.json'

vocabulary = json.load(open(vocab_path))
vectorizer = CountVectorizer(vocabulary=vocabulary)

def tobow(string):
    return vectorizer.transform([string]).toarray()