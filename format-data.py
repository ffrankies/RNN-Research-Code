###############################################################################
# Source: rnn tutorial from www.wildml.com
# This script closely follows the tutorial, repurposing it to work with python3.
# This part of the code creates a dataset from a given csv file. The csv file
# should contain only one column, start with a column heading, and contain
# text data in sentence format. The script will break the data down into
# sentences, tokenize them, and then saves the dataset in a file of the user's
# choice.
# The file will contain the following items, in the same order:
#   the vocabulary of the training set
#   the vector used to convert token indexes to words
#   the dictionary used to convert words to token indexes
#   the input for training, in tokenized format (as indexes)
#   the output for training, in tokenized format (as indexes)
#
# Author: Frank Wanye
# Date: 24 Nov 2016
###############################################################################
# Specify documentation format
__docformat__ = 'restructedtext en'

import _pickle as cPickle
import os
import numpy as np
import operator
import csv
import itertools
import nltk

# The number of words the RNN will recognize
vocabulary_size = 8000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

###############################################################################
# Reading the data file
###############################################################################
print("Reading the CSV data file...")

path = input("Enter path to the csv data file: ")
with open(path, "r", encoding='utf-8') as datafile:
    reader = csv.reader(datafile, skipinitialspace=True)
    reader.__next__() # Skips over table heading (contains 'body')

    print("Removing empty comments...")
    comments = []
    num = 0
    for item in reader:
        if not len(item) == 0:
            comments.append(item[0])
        num += 1
        if num % 10000 == 0:
            print("\rGone over %d comments." % num, end='')

print("\nBreaking comments down into sentences...")
sentences = itertools.chain(
    *[nltk.sent_tokenize(comment.lower()) for comment in comments])

print("Adding sentence start and end tokens to sentences...")
sentences = ["%s %s %s" %
    (sentence_start_token, sent, sentence_end_token) for sent in sentences]
print("Got %d sentences." % len(sentences))

print("Tokenizing sentences...")
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

print("Obtaining word frequency distribution...")
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print("Found %d unique words." % len(word_freq.items()))

# Building word-to-index and index-to-word vectors/dictionaries
vocabulary = word_freq.most_common(vocabulary_size - 1)
index_to_word = [word[0] for word in vocabulary]
index_to_word.append(unknown_token)
word_to_index = dict((word, index)
                     for index, word in enumerate(index_to_word))

print("Replacing all words not in vocabulary with unkown token...")
for index, sent in enumerate(tokenized_sentences):
    tokenized_sentences[index] = [
        word if word in word_to_index else unknown_token for word in sent]

print("Creating training data...")
x_train = np.asarray([[word_to_index[word] for word in sent[:-1]]
                     for sent in tokenized_sentences])
y_train = np.asarray([[word_to_index[word] for word in sent[1:]]
                     for sent in tokenized_sentences])

path = input("Enter the name of the file you wish to save the data as: ")
with open(path, "wb") as dataset_file:
    cPickle.dump((vocabulary, index_to_word, word_to_index, x_train, y_train),
                 dataset_file, protocol=2)
