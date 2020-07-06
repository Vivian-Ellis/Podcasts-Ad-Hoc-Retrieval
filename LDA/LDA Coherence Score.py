import joblib
from gensim.models import CoherenceModel

#load the model
lda_model=joblib.load('62topiclda.pkl')

#load the dictionary
dictionary=joblib.load('dictionary.pkl')

#load the corpus
bow_corpus=joblib.load('bow_corpus.pkl')

if __name__ == "__main__":
    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=bow_corpus.tolist(), dictionary=dictionary, coherence='c_v')

    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)
