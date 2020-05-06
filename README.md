# Project1
Analysing Youtube Video using WordCloud


import pandas as pd
import json
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from scipy.linalg import svd
import numpy as np

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
%matplotlib inline



import json

titles=[v['snippet']['title']for v in wl]
descriptions=[v['snippet']['description']for v in wl]

import csv

f=open('C:\\Users\\91949\\Desktop\\sub.txt') 
wl = json.load(f)
f.close()
wl[0]

titles=[v['snippet']['title']for v in wl]
descriptions=[v['snippet']['description']for v in wl]

type(wl)

from wordcloud import WordCloud
wl[0]

from wordcloud import WordCloud
wc=WordCloud().generate(" ".join(titles))
plt.figure(figsize=(10,12))
plt.imshow(wc)
plt.axis("off")

titles

descriptions
stopwords=['http','www','youtube','instagram','https','facebook','twitter']
stopwords+=list(ENGLISH_STOP_WORDS)
desc=' '.join(descriptions).lower()
wc=WordCloud(stopwords=stopwords,normalize_plurals=True).generate(desc)
plt.figure(figsize=(10,12))
plt.imshow(wc)
_=plt.axis('off')

ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
vect=TfidfVectorizer(stop_words=stopwords)
pip install sklearn
X=vect.fit_transform(titles)
X.shape
U,S,V=svd(X.todense(),full_matrices=False)
vocab=vect.get_feature_names()
def show_topics(a,n_words=5):
    top_words=lambda t:[vocab[i] for i in np.argsort(t)[:-n_words-1:1]]
    topic_words=([top_words(t) for t in a])
    return [''.join(t) for t in topic_words]
    
    
show_topics(V[:3])
from sklearn.decomposition import TruncatedSVD
tsne=TruncatedSVD(n_components=2)
x_red=tsne.fit_transform(X.todense())
plt.scatter(*x_red.T)
outliers=np.arange(x_red.shape[0])[x_red[:,1]<0.4]
plt.scatter(*x_red.T)
plt.scatter(*x_red[outliers,:].T,c="r")
outliers=[titles[i] for i in outliers]
wc=WordCloud(stopwords=stopwords,normalize_plurals=True).generate(" ".join(outliers).lower())
plt.figure(figsize=(10,12))
plt.imshow(wc)
_=plt.axis("off")

 
 
    
    
    


