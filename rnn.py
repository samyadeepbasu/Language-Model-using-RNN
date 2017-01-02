########### Language Model using Recurrent Neural Networks ##############

import csv
import itertools
import sys
import os
import time
import nltk
import numpy as np 
import operator
from utils import *

vocabulary_size = 8000 
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"
unknown_token = "UNKNOWN_TOKEN"

with open('data/reddit-comments-2015-08.csv','rb') as f:
	#Reading the entire row of comments at a once
	reader = csv.reader(f,skipinitialspace=True)
	#Skipping the first row
	reader.next()
	#Splitting the comments into full-sentences for further processing
	temp = [nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader]
	#Creates a single object from all the elements of the lists of lists
	sentences = itertools.chain(*temp)
	#Appending sentence_first_token and sentence_last_token
	sentences = ["%s %s %s" % (sentence_start_token,x,sentence_end_token) for x in sentences]

#End of File Parsing

#Tokenizing the sentences into words
tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

#Counting the frequencies of the word
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))

unique_word_count = len(word_freq.items())

#Extracting the most common words and building the index
vocab = word_freq.most_common(vocabulary_size-1)
#Converting the object to a list with the most frequent words
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)

word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

#Replacement of words not in the vocabulary with unknown token
for i,sent in enumerate(tokenized_sentences):
	tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]


#Creating the training data array
X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

#END OF GENERATION OF DATA for the network





