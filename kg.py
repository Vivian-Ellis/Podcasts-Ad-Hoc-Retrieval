import re
import pandas as pd
import bs4
import requests
import gensim
import spacy
import sys, os
import json
import sklearn
import spacy
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet
from pathlib import Path
nlp = spacy.load("en_core_web_lg")
# nlp = spacy.load("en")
sys.path.append('C:/../Spotify/')
import querysemantic as q
from spacy import displacy

def main(query):
    doc = nlp(query)
    svg=displacy.render(doc, style="dep")
    # # svg=displacy.serve(doc, style="dep")
    output_path = Path(os.path.join("./", "sentence.html"))
    output_path.open('w', encoding="utf-8").write(svg)
    # print(q.main(query))
    synonyms = []
    #expand via wordnet
    interesting_words=[]
    for i in doc:
        if i.pos_ in ["NOUN"]:
            comps = [j for j in i.children if j.dep_ == "compound"]
            if comps:
                interesting_words.append(str(comps[0])+'_'+str(i))
                synonyms.append(str(comps[0])+' '+str(i))
    # print('interesting words: ',interesting_words)

    for word in interesting_words:
        for syn in wordnet.synsets(word):
            # print(syn)
            for l in syn.lemmas():
                synonyms.append(l.name().replace('_',' '))
                # docl=nlp(l.name())
            for hypo in syn.hyponyms():
                # print(hypo)
                for h in hypo.lemma_names():
                    doch=nlp(h)
                    if doch.vector_norm:
                        if doc.similarity(doch) > .7:
                            synonyms.append(h.replace('_',' '))
            for hype in syn.hypernyms():
                # print(hype)
                for e in hype.lemma_names():
                    doce=nlp(e)
                    if doce.vector_norm:
                        if doc.similarity(doce) > .7:
                            synonyms.append(e.replace('_',' '))


    # print('Becoming in gkg: ',q.main("Becoming"))

    for chunk in doc:
        # print(chunk.text, chunk.lemma_, chunk.pos_, chunk.tag_, chunk.dep_,
        #     chunk.shape_, chunk.is_alpha, chunk.is_stop)
        # if chunk.pos_=='DET' and chunk.dep_=='ROOT':
        #     synonyms.append(q.main(chunk.text))
        # chunk.dep_=='dobj' or chunk.dep_=='obj' or chunk.dep_=='pobj' or
        if chunk.pos_ =='NOUN':
            synonyms.append(chunk.text)
            doc3=nlp(chunk.text)

            # print('wordnet: ',subject_wordnet(ent))
            for syn in wordnet.synsets(chunk.text):
                # print('syn lemma: ',syn.lemmas()[0])

                # for l in syn.lemmas():
                #     synonyms.append(l.name())
                #     docl=nlp(l.name())
                #     if docl.vector_norm:
                #         if doc3.similarity(docl) > .7:
                #             synonyms.append(l.name())
                        # print('l: ',l.name(),' sim: ',doc3.similarity(docl))
                for hypo in syn.hyponyms():
                    for h in hypo.lemma_names():
                        doch=nlp(h)
                        if doch.vector_norm:
                            if doc3.similarity(doch) > .7:
                                synonyms.append(h)
                for hype in syn.hypernyms():
                    for e in hype.lemma_names():
                        doce=nlp(e)
                        if doce.vector_norm:
                            if doc3.similarity(doce) > .7:
                                synonyms.append(e)
    # # gkg=[]
    # #expand via gkg

    for ent in doc.ents:
        # print('ent: ',ent)
        # print('ent label: ',ent.label_)
        if ent.label_ =='PERSON' or ent.label_ =='NORP' or ent.label_=='FAC' or ent.label_=='ORG' or ent.label_=='LOC' or ent.label_ =='PRODUCT' or ent.label_=='EVENT' or ent.label_=='WORK_OF_ART' or ent.label_=='LAW':
            synonyms.append(q.main(ent.text))
        if ent.label_=='GPE':
            synonyms.append(ent.text)

    # synonyms.append(interesting_words)

    # print('results: ',synonyms)

    # # for k in synonyms:
    #     # for g in k:
    #     doc2=nlp(k)
    #     if doc.similarity(doc2) > .3:
    #         gkg.append(k)

    # print('synonyms: ',synonyms)
    # synonyms.append("becoming Book by Michelle Obama memoir unit state ladi michell obama publish describ author deepli person experi book talk root voic time white hous public health campaign role mother")
    return ' '.join(synonyms)

if __name__=="__main__":
    # query="What were people saying about the spread of the novel coronavirus NCOV-19 in Wuhan at the end of 2019? Also what are people saying about a vaccine?"
    query=sys.argsv[1:]

    # query="Anna Sorokina moved to New York City in 2013 and posed as wealthy German heiress Anna Delvey.  In 2019 she was convicted of grand larceny, theft, and fraud.  What were people saying about her, the charges, her trial, and New York socialite society in general?"
    # query="Greta Thunberg crossed the Atlantic Ocean"
    # query="How was Greta Thunberg’s sailing trip across the Atlantic Ocean related to global climate change?"
    # query="Becoming by Michelle obama"
    # query="Former First Lady Michelle Obama’s memoir Becoming was published in early 2019.  What were people saying about it?"
    main(query)
