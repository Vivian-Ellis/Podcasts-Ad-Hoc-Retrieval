#class to create the lda training model and save to pickle for later reference
import sklearn
import gensim
import numpy as np
import nltk
import pandas as pd
import mysql.connector
import joblib
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import pickle
import traceback
from gensim.models import CoherenceModel

con = mysql.connector.connect(user='root', password='root',
                              host='127.0.0.1',
                              database='lda_db',
                              allow_local_infile = "True")
cursor = con.cursor(buffered=True)
query = ("select transcript from training")
#get list of transcripts
cursor.execute(query)
lda_train=cursor.fetchall()
cursor.close()
con.close()

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

for doc in lda_train:
	processed_docs.append(preprocess(doc[0]))

#create a dictionary from processed_docs containing the number of times a word appears in the training set using gensim.corpora.Dictionary and call it dictionary
dictionary=gensim.corpora.Dictionary(processed_docs)

#remove very rare and very common words
dictionary.filter_extremes(no_below=15, no_above=0.1, keep_n= 100000)

#pickle the dictionary for later use
dictionary.save("C:/Users/15099/Documents/Projects/Spotify/LDA/dictionary.pkl")

#create the bag-of-words model for each document
bow_corpus=[dictionary.doc2bow(doc) for doc in processed_docs]

#pickle the bow corpus
pickle.dump(bow_corpus, open('C:/Users/15099/Documents/Projects/Spotify/LDA/bow_corpus.pkl', 'wb'))

#train LDA model
lda_model = gensim.models.LdaModel(bow_corpus,num_topics = 62,id2word = dictionary,passes = 30)

#explore words in topics
for idx, topic in lda_model.print_topics(-1):
    print("Topic: {} \nWords: {}".format(idx, topic ))
    print("\n")

#pickle the model in a file
joblib.dump(lda_model,'C:/Users/15099/Documents/Projects/Spotify/LDA/62topiclda.pkl')

# if __name__ == "__main__":
#     # Compute Coherence Score
#     coherence_model_lda = CoherenceModel(model=lda_model, texts=processed_docs, dictionary=dictionary, coherence='c_v')
#
#     coherence_score = coherence_model_lda.get_coherence()
#     print('\nCoherence Score: ', coherence_score)
