import spacy
import re
import numpy as np
from gensim.utils import simple_preprocess

nlp = spacy.load("en_core_web_sm")





# -------------Cleaning text and transforme to tokenze ---------------------------------------------

def clean_text(text):
    return [simple_preprocess(pattern) for pattern in text]

# -------------transforme the tokenz to vectors numerical ---------------------------------------------


def sentence_to_vector(tokens, model):
    vecs = [model.wv[word] for word in tokens if word in model.wv]
    if len(vecs) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vecs, axis=0)






# Fonction pour convertir une phrase en vecteur
def sentence_to_vector(tokens, model):
    vecs = [model.wv[word] for word in tokens if word in model.wv]
    if len(vecs) == 0:
        return np.zeros(model.vector_size)  # Si aucun mot n'est trouvé, retourner un vecteur de zéros
    return np.mean(vecs, axis=0) 