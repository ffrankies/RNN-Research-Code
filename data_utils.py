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
# Date: 21 Feb 2017
###############################################################################

# Specify documentation format
__docformat__ = 'restructedtext en'

try:
    import _pickle as cPickle
except Exception:
    import cPickle
import os
import numpy as np
import operator
import csv
import itertools
import nltk
import logging
import logging.handlers
import argparse
import time

###############################################################################
# Setting up global variables
###############################################################################
log = None # Logging
timestr = time.strftime("%d%m%y%H") # For when current time is needed
comments = [] # Holds the comments
sentences = [] # Holds the sentences from the comments
vocab_size = 0 # Number of words RNN wil remember
# Special tokens
unknown = "UNKNOWN_TOKEN"
sentence_start = "SENTENCE_START"
sentence_end = "SENTENCE_END"
paragraph_start = "PARAGRAPH_START" # Use split on (\n or \n\n) for this?
paragraph_end = "PARAGRAPH_END"
story_start = "STORY_START"
story_end = "STORY_END"

def createDir(dirPath):
    """
    Creates a directory if it does not exist.

    :type dirPath: string
    :param dirPath: the path of the directory to be created.
    """
    try:
        os.makedirs(dirPath, exist_ok=True) # Python 3.2+
    except TypeError:
        try: # Python 3.2-
            os.makedirs(dirPath)
        except OSError as exception:
            if exception.errno != 17:
                raise
# End of createDir()

def set_up_logging(name='DATA', dir='logging'):
    """
    Sets up logging for the data formatting.

    :type name: String
    :param name: the name of the logger. Defaults to 'DATA'

    :type dir: String
    :param dir: the directory in which the logging will be done. Defaults to
                'logging'
    """
    createDir(dir) # Create log directory in system if it isn't already there
    global log
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)

    handler = logging.handlers.RotatingFileHandler(
        filename=dir+"/data.log",
        maxBytes=1024*512,
        backupCount=5
    )

    formatter = logging.Formatter(
        "%(asctime)s-%(name)s-%(levelname)s-%(message)s"
    )

    handler.setFormatter(formatter)

    log.addHandler(handler)
    log.info("Set up logging for data formatting session.")
# End of set_up_logging()

def read_csv(path=None):
    """
    Reads the given csv file and extracts data from it into the comments array.
    Empty data cells are not included in the output.

    :type path: String
    :param path: the path to the csv data file
    """
    global log
    global comments

    if path is None:
        path = input("Enter path to the scv data file: ")

    log.info("Reading the csv data file at: %s" % path)
    with open(path, "r", encoding='utf-8') as datafile:
        reader = csv.reader(datafile, skipinitialspace=True)
        reader.__next__() # Skips over table heading
        num_seen = 0
        num_saved = 0
        for item in reader:
            if len(item) > 0 and len(item[0]) > 0:
                comments.append(item[0])
                num_saved += 1
            num_seen += 1
        log.info("Gone over %d examples, saved %d of them" %
                 (num_seen, num_saved))
# End of read_csv()

def tokenize_sentences():
    """
    Uses the nltk library to break comments down into sentences, and then
    tokenizes the words in the sentences. Also appends the sentence start and
    end tokens to each sentence.
    """
    global sentences
    global comments
    global sentence_start
    global sentence_end
    global log

    log.info("Breaking comments down into sentences.")
    sentences = itertools.chain(
        *[nltk.sent_tokenize(comment.lower()) for comment in comments])
    log.info("%d sentences found in dataset." % len(sentences))

    log.info("Adding sentence start and end tokens to sentences.")
    sentences = ["%s %s %s" % (sentence_start, sentence, sentence_end)
                 for sentence in sentences]

    log.info("Tokenizing words in sentences.")
    sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
# End of tokenize_sentences()

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
