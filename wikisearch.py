import wikipedia as wiki
import sklearn
import gensim
import numpy as np
import nltk
import pandas as pd
import joblib
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *

summary=wiki.summary("Babe Ruth").lower()
print(summary)
print("\n")

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

processed_adhoc=preprocess(summary)

print(processed_adhoc)

#load the model
lda_model=joblib.load('62topiclda.pkl')

#create a dictionary from processed_adhoc containing the number of times a word appears
dictionary=gensim.corpora.Dictionary([processed_adhoc])
lda_model.id2word=dictionary

#create bow vector
bow_vector=dictionary.doc2bow(processed_adhoc)

results = lda_model.get_document_topics(bow_vector,minimum_probability=.13)
print(results)
print('\n')
