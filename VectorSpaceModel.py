#class to create the lda training model and save to pickle for later reference
import sklearn
import gensim
import numpy as np
import nltk
import collections
import pandas as pd
import mysql.connector
import itertools
import joblib
import json
import sys,os
sys.path.append('C:/../Spotify/')
import preprocess as p
from sklearn.metrics.pairwise import linear_kernel
from gensim import models
from collections import OrderedDict
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import similarities
from sklearn.metrics.pairwise import cosine_similarity

def mysql_full_transcript(files):
    con = mysql.connector.connect(user='root', password='root',
                                  host='127.0.0.1',
                                  database='episodes',
                                  allow_local_infile = "True")
    cursor = con.cursor()
    full_transcript=[]
    for file in files:
        query="select preprocessed_transcript from full_transcript where episode_uri in ('"+file+"')"
        # print(query)
        cursor.execute(query)
        full_transcript.append(list(cursor.fetchone()))
        con.commit()
    #close mysql connection
    cursor.close()
    con.close()
    full_transcript=list(itertools.chain.from_iterable(full_transcript))
    return full_transcript

def mysql_interval(files):
    con = mysql.connector.connect(user='root', password='root',
                                  host='127.0.0.1',
                                  database='episodes',
                                  allow_local_infile = "True")
    cursor = con.cursor(buffered=True)
    interval_transcript=[]
    for file in files:
        query="select preprocessed_transcript_chunk from 30_sec_intervals where episode_uri in ('"+file+"')"
        # print(query)
        cursor.execute(query)
        interval_transcript.append(list(cursor.fetchone()))
        con.commit()
    #close mysql connection
    cursor.close()
    con.close()
    interval_transcript=list(itertools.chain.from_iterable(interval_transcript))
    return interval_transcript

#takes a list of json file names and extracts
#full transcript and populates the corpus
#example
#['083Gl9wcuPf4TTts92G87d.json', '10hwGfpcijREIiz4KX7WqE.json']
def build_corpus_episodes(files):
    corpus=[]
    for file in files:
        filename = os.fsdecode('C:/Users/15099/Documents/Projects/Spotify/dataset/test/'+file)
        # Opening JSON file
        jsonfile = open(filename,)
        # returns JSON object as a dictionary
        data = json.load(jsonfile)
        full_transcript=""
        # Iterating through the json list
        for results in data['results']:
            for alternatives in results['alternatives']:
                #if the transcript is not null
                if alternatives.get('transcript'):
                    full_transcript+=alternatives.get('transcript')
        corpus.append(full_transcript)
    return corpus

def build_corpus_chunks(files):
    thirty_second_chunks={'transcript':[],'startTime':[],'file_Name':[]}
    for file in files:
        filename = os.fsdecode('C:/Users/15099/Documents/Projects/Spotify/dataset/test/'+file)
        # Opening JSON file
        jsonfile = open(filename,)
        # returns JSON object as a dictionary
        data = json.load(jsonfile)
        # Iterating through the json list
        for results in data['results']:
            for alternatives in results['alternatives']:
                if alternatives.get('transcript'):
                    transcript=' '.join(p.preprocess(alternatives.get('transcript')))
                    thirty_second_chunks['transcript'].append(transcript)
                    thirty_second_chunks['startTime'].append(alternatives.get('words')[0].get('startTime'))
                    thirty_second_chunks['file_Name'].append(file)
    return thirty_second_chunks

#stem, tokenize and lemmatize corpus
def preprocess_corpus(corpus):
    processed_docs=[]
    for doc in corpus:
        doc_text=p.preprocess(doc)
        doc_text = ' '.join(doc_text)
        processed_docs.append(doc_text)
    return processed_docs

#stem, tokenize and lemmatize query
def preprocess_query(query):
    processed_query=p.preprocess(query)
    processed_query=[' '.join(processed_query)]
    return processed_query

def create_corpus_tfidf(corpus,vectorizer):
    corpus_vector=vectorizer.fit_transform(preprocess_corpus(corpus))
    return corpus_vector

def create_query_vector(query,vectorizer):
    query_vector=vectorizer.transform(query)
    return query_vector

#calculate the cosine  similarity between the corpus and query
def cos_sim(corpus_vector,query_vector,sim):
    cosine_similarities = cosine_similarity(corpus_vector,query_vector)
    similarity_indices=[]
    for i in range(len(cosine_similarities)):
        # print("episode has a cos sim of ",cosine_similarities[i],": ",i)
        if cosine_similarities[i] > sim:
            # print("episode has a cos sim of ",cosine_similarities[i],": ",i)
            similarity_indices.append(i)

    return similarity_indices

#calculate the cosine  similarity between the corpus and query
def cos_sim_ordered(corpus_vector,query_vector,sim):
    cosine_similarities = cosine_similarity(corpus_vector,query_vector)
    similarity_indices=[]
    temp={}
    for i in range(len(cosine_similarities)):
        if cosine_similarities[i] > sim:
            # print("episode has a cos sim of ",cosine_similarities[i],": ",i)
            temp[cosine_similarities[i][0]]=i

    od=collections.OrderedDict(sorted(temp.items(),reverse=True))
    # print(temp)
    counter=0
    for key,value in od.items():
        similarity_indices.append(value)
        counter+=1
        if counter >= 10:
            return similarity_indices
    return similarity_indices

def main(query,layer,files):
    # query="trump presidency end"
    vectorizer=TfidfVectorizer()

    #layer 1----------------------------------------------
    if layer==1:
        # print('vsm query: ',query)
        corpus=build_corpus_episodes(files)
        # print(corpus)
        corpus_vector=create_corpus_tfidf(corpus,vectorizer)
        query_vector=create_query_vector(query,vectorizer)
        # print(query_vector)
        # print("--------")
        indices=cos_sim(corpus_vector,query_vector,.1)
        layertwofiles=[]
        for i in indices:
            layertwofiles.append(files[i])
        # print('first layer filtered to: ',layertwofiles)
        return layertwofiles

    #layer 2-----------------------------------------------
    if layer ==2:
        intervals=build_corpus_chunks(files)
        # print(intervals.get('transcript'))
        intervals_vector=create_corpus_tfidf(intervals.get('transcript'),vectorizer)
        # print(intervals_vector)
        # print("----------")
        query_vector=create_query_vector(query,vectorizer)
        # print(query_vector)
        results=cos_sim_ordered(intervals_vector,query_vector,.2)
        # print(results)
        relevant_segments={'file_Name':[],'startTime':[]}
        for j in results:
            # relevant_segments['transcript'].append(intervals.get('transcript')[j])
            relevant_segments['file_Name'].append(intervals.get('file_Name')[j])
            relevant_segments['startTime'].append(intervals.get('startTime')[j])
        return relevant_segments

if __name__=="__main__":
    query, layer, files=sys.argsv[1:]
    main(query, layer,files)
    # main(argsv[1:])
