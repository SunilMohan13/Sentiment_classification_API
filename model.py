# ML imports
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

import pickle

from util import plot_roc

class NLPModel(object):

	def __init__(self):
		'''Simple NLP
		Attributes:
			clf: sklearn classifier model
			vectorizer: TFIDF vectorizer or similar
		'''
		self.clf = MultinomialNB()
		self.vectorizer = TfidfVectorizer()
		
	
	def vectorizer_fit(self, X):
		'''Fits a TFIDF vectorizer to the text'''
		self.vectorizer.fit(X)
		
	def vectorizer_transform(self, X):
		'''Transform the text data to a sparse TFIDF matrix'''
		X_transformed = self.vectorizer.transform(X)
		return X_transformed
		
	def train(self, X, y):
		'''Trains the classifier to associate the label with the sparse matrix'''
		self.clf.fit(X, y)
		
	def predict_proba(self, X):
		'''Returns probability for the binary class '1' in a numpy array'''
		y_proba = self.clf.predict_proba(X)
		return y_proba[:,1]
		
	def predict(self, X):
		'''Returns the predicted class in a array'''
		y_pred = self.clf.predict(X)
		return y_pred
		
	def pickle_vectorizer(self, path='lib/models/TFIDFVectorizer.pkl'):
		'''Saves the trained vectorizer for future use'''
		with open(path, 'wb') as f:
			pickle.dump(self.vectorizer, f)
			print("Pickled vectorizer at {}".format(path))
			
	def pickle_clf(self, path='lib/models/SentimentClassifier.pkl'):
		'''Saves the trained classifier for future use'''
		with open(path, 'wb') as f:
			pickle.dump(self.clf, f)
			print("Pickled classifier at {}".format(path))
			
	def plot_roc(self, X, y, size_x, size_y):
		'''Plot the ROC curve for X_test and y_test'''
		plot_roc(self.clf, X, y, size_x, size_y)