import sklearn
import gensim
import numpy as np
import nltk
import pandas as pd

#example dataset
from sklearn.datasets import fetch_20newsgroups
from sklearn.externals import joblib
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *

newsgroups_train=fetch_20newsgroups(subset='train',shuffle=True)
newsgroups_test=fetch_20newsgroups(subset='test',shuffle=True)
np.random.seed(400)

#stemmer
stemmer = SnowballStemmer("english")

def lemmatize_stemming(text):
	return stemmer.stem(WordNetLemmatizer().lemmatize(text,pos='v'))

#tokenize and lemmatize
def preprocess(text):
	result=[]
	for token in gensim.utils.simple_preprocess(text):
		if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
			result.append(lemmatize_stemming(token))
	return result

processed_docs=[]

for doc in newsgroups_train.data:
	processed_docs.append(preprocess(doc))
	
#create a dictionary from processed_docs containing the number of times a word appears in the training set using gensim.corpora.Dictionary and call it dictionary
dictionary=gensim.corpora.Dictionary(processed_docs)

#create the bag-of-words model for each document 
bow_corpus=[dictionary.doc2bow(doc) for doc in processed_docs]
	
#test model on unseen doc
num=100
unseen_document = newsgroups_test.data[num]
print(unseen_document)
#data preprocessing step for the unseen doc
bow_vector=dictionary.doc2bow(preprocess(unseen_document))

#load the model
lda_from_joblib=joblib.load('examplelda.pkl')

#use the loaded model
for index, score in sorted(lda_from_joblib[bow_vector],key=lambda tup: -1*tup[1]):
	print("Score: {}\t Topic: {}".format(score,lda_from_joblib.print_topic(index,5)))
	
print(newsgroups_test.target[num])