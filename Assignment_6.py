#importing the required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os                                                       
import re                                                       
import random                                                   
import scipy as sp
import numpy as np
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize                         
from nltk.corpus import stopwords                               
from nltk.stem.porter import *                                  
from nltk.stem.snowball import SnowballStemmer                  
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer 
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
from nltk.util import ngrams
from nltk import RegexpParser

categories=['comp.graphics','rec.autos','sci.space']

datadir = "20_newsgroups"
os.path.join(datadir, categories[0])

datadir = "20_newsgroups"

paths=[]
l=[]

for category in categories:
    path = os.path.join(datadir, category)
    paths.append(path)

for path in paths:
    for i in range(1):
        choice = random.choice(os.listdir(path)) 
        fullfilename = os.path.join(path, choice)
        with open(fullfilename) as f:
            l.append(f.read().splitlines())

l

#converting into flat_list
lines = [] 
for sublist in l:
    for item in sublist:
        lines.append(item)

lines

import nltk
nltk.download('words')

import nltk
nltk.download('averaged_perceptron_tagger')

import nltk
nltk.download('stopwords')

import nltk
nltk.download('wordnet')


corpus=[]
# Generating n-grams from sentences.
def extract_ngrams(line, num):
    n_grams = ngrams(line, num)
    return [ ' '.join(grams) for grams in n_grams]



words = set(nltk.corpus.words.words())
lemmatizer = WordNetLemmatizer() 
from nltk import word_tokenize, pos_tag
for i in range(len(lines)):
    line=[]
    line=re.sub('[^a-zA-Z]',' ',lines[i])
    line = line.lower()
    line = line.split()#tokenization 
    print("tokenized words")
    print(line)
    print("\n")
    line=[word for word in line if word not in set(stopwords.words('english'))]
    print("Removing Stopwords")
    print(line)
    print('\n')
    line = [lemmatizer.lemmatize(word) for word in line]
    print("Lemmatization")
    print(line)
    print('\n')
    line = [ps.stem(word) for word in line]
    print("Stemming")
    print(line)
    print('\n')
    print("pos_tags and chunking")
    print("After Split:",line)
    tokens_tag = pos_tag(line)
    print("After pos_tags:",tokens_tag)
    patterns= """mychunk:{<NN.?>*<VBD.?>*<JJ.?>*<CC>?}"""
    chunker = RegexpParser(patterns)
    print("After Regex:",chunker)
    output = chunker.parse(tokens_tag)
    print("After Chunking",output)
    print('\n')
    
    
    
    print(line)
    print("N_grams")
    print("2-gram: ", extract_ngrams(line, 2))
    print('\n')
    
    line= [word for word in line if word in words]#taking dictionary words only to build final bag of word models
    line = [word for word in line if len(word)>1]



    line=' '.join(line)
    if(line):
        corpus.append(line)


corpus

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

cats=['comp.graphics','talk.religion.misc','sci.space','alt.atheism']

newsgroups_train = fetch_20newsgroups(subset = 'train',categories = cats)

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(newsgroups_train.data)
vectors.shape

newsgroups_train.filenames.shape

newsgroups_train.target.shape

newsgroups_train.target[:10]

vectors.nnz / float(vectors.shape[0])

from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
newsgroups_test = fetch_20newsgroups(subset='test',categories = cats)

vectors_test = vectorizer.transform(newsgroups_test.data)

clf = MultinomialNB(alpha = .01)

clf.fit(vectors,newsgroups_train.target)

pred = clf.predict(vectors_test)
metrics.f1_score(newsgroups_test.target,pred,average = 'macro')

import numpy as np
def show_top10(classifier, vectorizer, categories):
    feature_names = np.asarray(vectorizer.get_feature_names())
    for i, category in enumerate(categories):
        top10 = np.argsort(classifier.coef_[i])[-10:]
        print("%s: %s" % (category, " ".join(feature_names[top10])))
        
show_top10(clf, vectorizer, newsgroups_train.target_names)

newsgroups_test = fetch_20newsgroups(subset='test',remove=('headers', 'footers', 'quotes'),categories = cats)
vectors_test = vectorizer.transform(newsgroups_test.data)
pred = clf.predict(vectors_test)
metrics.f1_score(pred, newsgroups_test.target, average='macro')

newsgroups_train = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes'),categories=categories)
vectors = vectorizer.fit_transform(newsgroups_train.data)
clf = MultinomialNB(alpha=.01)
clf.fit(vectors, newsgroups_train.target)
vectors_test = vectorizer.transform(newsgroups_test.data)
pred = clf.predict(vectors_test)
metrics.f1_score(newsgroups_test.target, pred, average='macro')

