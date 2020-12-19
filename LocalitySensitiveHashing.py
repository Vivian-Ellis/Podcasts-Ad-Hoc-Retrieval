import numpy as np
import pandas as pd
import re
import gensim
import time
import nltk
import json
import sys, os
sys.path.append('C:/../Spotify/')
import preprocess as p
from datasketch import MinHash, MinHashLSHForest
from nltk.stem.porter import *


def build_corpus_episodes(files):
    corpus=[]
    for file in files:
        filename = os.fsdecode('C:/..'+file)
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

#stem, tokenize and lemmatize corpus
def preprocess_corpus(corpus):
    processed_docs=[]
    for doc in corpus:
        doc_text=p.preprocess(doc)
        doc_text = ' '.join(doc_text)
        processed_docs.append(doc_text)
    return processed_docs

def get_forest(data, perms):
    start_time=time.time()

    minhash=[]

    for text in data:
        tokens = p.preprocess(text)
        m=MinHash(num_perm=perms)
        for s in tokens:
            m.update(s.encode('utf-8'))
        minhash.append(m)

    forest = MinHashLSHForest(num_perm=perms)

    for i,m in enumerate(minhash):
        forest.add(i,m)

    forest.index()

    print('time to build forest: ',(time.time()-start_time))

    return forest

def predict(tokens, database, perms, num_results, forest):
    start_time=time.time()

    m = MinHash(num_perm=perms)
    for s in tokens:
        m.update(s.encode('utf-8'))

        idx_array=np.array(forest.query(m,num_results))
        if len(idx_array)==0:
            return None

        # print(idx_array)
        # result=database[idx_array]

        print('took % seconds to query forest'%(time.time()-start_time))

        return idx_array

def build_corpus(files):
    thirty_second_chunks={'transcript':[],'startTime':[],'file_Name':[]}
    for file in files:
        filename = os.fsdecode('C:/../'+file)
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

def main(query,files):
    # all_thirty_sec_chunks = build_corpus(args)
    # corpus=all_thirty_sec_chunks.get('transcript')
    corpus=preprocess_corpus(build_corpus_episodes(files))
    # print(corpus)
    permutations=128
    num_recommendations=5
    forest=get_forest(corpus,permutations)
    # query="coping with tragedy"
    # processed_adhoc=p.preprocess(query)
    # processed_adhoc=[' '.join(processed_adhoc)]
    # print(processed_adhoc)
    result = predict(query,corpus,permutations,num_recommendations, forest)
    print('top result: ',result)

if __name__ == '__main__':
    # main(['00epqmeUGZodleWzOysye4.json'])
    # main(argsv[1:])
    query, files=sys.argsv[1:]
    main(query,files)
