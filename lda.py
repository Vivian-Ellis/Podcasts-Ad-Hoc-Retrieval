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

#LDA
lda_model = gensim.models.LdaModel(bow_corpus,num_topics = 8,id2word = dictionary,passes = 20)

#save the model as a pickle in a file
joblib.dump(lda_model,'examplelda.pkl')