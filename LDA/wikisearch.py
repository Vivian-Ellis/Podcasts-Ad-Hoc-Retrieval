#This code will take a user ad-hoc and search for a description of the ad-hoc.
#The LDA model is then loaded and classifies the ad-hoc into a topic.
import wikipedia as wiki
import sklearn
import gensim
import numpy as np
import nltk
import pandas as pd
import urllib.request
import joblib
from googlesearch import search
from bs4 import BeautifulSoup
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *

adhoc="babe ruth"
summary=""

try: #first search wiki
    summary=wiki.summary(adhoc)
except: #no wiki results grab top 5 google articles
    query =adhoc
    summary=adhoc
    results = []

    for i in search(query,tld='com',lang='en',num=5, stop=5, pause=2):
        results.append(i)

    for link in results:
        try:
            html = urllib.request.urlopen(link)
        except Exception as e:
            pass
        else:
            # get the html contents of the url
            html_contents = html.read()#.decode(html.headers.get_content_charset())#html.read()
            #make the soup
            soup=BeautifulSoup(html_contents,'lxml')
            summary +=' '+soup.text
    #print(summary.encode("utf-8"))

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

# print(processed_adhoc)

#load the model
lda_model=joblib.load('C:/Users/15099/Documents/Projects/Spotify/LDA/62topiclda.pkl')

#load the dictionary
dictionary=joblib.load('C:/Users/15099/Documents/Projects/Spotify/LDA/dictionary.pkl')

#create bow vector
bow_vector=dictionary.doc2bow(processed_adhoc)

results = lda_model.get_document_topics(bow_vector,minimum_probability=.15)
print("Topic Distribution: ")
print(results)

#explore words in topics
for idx, topic in lda_model.print_topics(-1):
    print("Topic: {} \nWords: {}".format(idx, topic ))
    print("\n")
