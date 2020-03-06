# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 12:09:22 2020

@author: geeta
"""

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('wordnet')

sentence = open('input.txt', encoding="utf8").read()

# Tokenization
stokens = nltk.sent_tokenize(sentence)
wtokens = nltk.word_tokenize(sentence)

print("\nWord  Tokenization:\n")
print(wtokens)
print("\nSentence  Tokenization:\n")
print(stokens)


# Stemming
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import SnowballStemmer
print("\nStremming:\n")
pStemmer = PorterStemmer()
lStemmer = LancasterStemmer()
sStemmer = SnowballStemmer('english')

n1 = 0
for t in wtokens:
    n1 = n1 + 1
    if n1 < 4:
        print(pStemmer.stem(t), lStemmer.stem(t), sStemmer.stem(t))



# POS
# Lemmatization
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
print("\nPOS / Lemmatization\n")
n1 = 0
for t in wtokens:
    n1 = n1 + 1
    if n1 < 6:
        print("Lemmatizer:", lemmatizer.lemmatize(t), ",    With POS=a:", lemmatizer.lemmatize(t, pos="a"))



# Trigram
from nltk.util import ngrams
print("\nTrigram\n")
n = 0
for s in stokens:
    n = n + 1
    if n < 2:
        token = nltk.word_tokenize(s)
        bigrams = list(ngrams(token, 2))
        trigrams = list(ngrams(token, 3))
        print("The text:", s, "\nword_tokenize:", token, "\nbigrams:", bigrams, "\ntrigrams", trigrams)


# Named Entity Recognition
from nltk import word_tokenize, pos_tag, ne_chunk
print("\nNamed Entity Recognition:\n")
n = 0
for s in stokens:
    n = n + 1
    if n < 2:
        print(ne_chunk(pos_tag(word_tokenize(s))))