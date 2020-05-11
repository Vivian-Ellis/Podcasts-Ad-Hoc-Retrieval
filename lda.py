import sklearn
import gensim
import numpy as np
import nltk
import pandas as pd

#example dataset
from sklearn.datasets import fetch_20newsgroups
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *

newsgroups_train=fetch_20newsgroups(subset='train',shuffle=True)
newsgroups_test=fetch_20newsgroups(subset='test',shuffle=True)
np.random.seed(400)

#nltk.download('wordnet')


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
	
#print(processed_docs[:2])
	
#create a dictionary from processed_docs containing the number of times a word appears in the training set using gensim.corpora.Dictionary and call it dictionary
dictionary=gensim.corpora.Dictionary(processed_docs)

#checking dictionary created
# count=0
# for k,v in dictionary.iteritems():
	# #print(k,v)
	# count +=1
	# if count >10:
		# break

#optional step to remove very rare and very common words
#dictionary.filter_extremes(no_below=15,no_above=.1,keep_n=100000)

#create the bag-of-words model for each document 
bow_corpus=[dictionary.doc2bow(doc) for doc in processed_docs]

#preview BOW for our sample preprocessed document
#document_num=20
#bow_doc_x=bow_corpus[document_num]

#for i in range(len(bow_doc_x)):
#	print("Word {} (\"{}\") appears {} time.".format(bow_doc_x[i][0],dictionary[bow_doc_x[i][0]],bow_doc_x[i][1]))

#LDA
lda_model = gensim.models.LdaModel(bow_corpus,num_topics = 8,id2word = dictionary,passes = 20)

#explore the words occurring and its relative weight 
# for idx, topic in lda_model.print_topics(-1):
    # print("Topic: {} \nWords: {}".format(idx, topic ))
    # print("\n")
	
#test model on unseen doc
num=100
unseen_document = newsgroups_test.data[num]
print(unseen_document)
#data preprocessing step for the unseen doc
bow_vector=dictionary.doc2bow(preprocess(unseen_document))

for index, score in sorted(lda_model[bow_vector],key=lambda tup: -1*tup[1]):
	print("Score: {}\t Topic: {}".format(score,lda_model.print_topic(index,5)))
	
print(newsgroups_test.target[num])