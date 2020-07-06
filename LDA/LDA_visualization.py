#Build a visualization of the trained LDA model
from googlesearch import search
import requests
import pyLDAvis.gensim
import pandas as pd
import joblib
import os

#load the model
lda_model=joblib.load('C:/Users/15099/Documents/Projects/Spotify/LDA/62topiclda.pkl')

#load the dictionary
dictionary=joblib.load('C:/Users/15099/Documents/Projects/Spotify/LDA/dictionary.pkl')

#load the corpus
bow_corpus=joblib.load('C:/Users/15099/Documents/Projects/Spotify/LDA/bow_corpus.pkl')

vis = pyLDAvis.gensim.prepare(lda_model, bow_corpus, dictionary)

pyLDAvis.show(vis)

pyLDAvis.save_html(vis,'LDA_vis.html')
