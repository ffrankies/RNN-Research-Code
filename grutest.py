###############################################################################
# Tests the Vanilla RNN on a small dataset
###############################################################################

import grurnn as RNN

# Specify documentation format
__docformat__ = 'restructedtext en'

try:
    import _pickle as cPickle
except Exception:
    import cPickle
import theano
import theano.tensor as T
import numpy as np
import os
import sys
import time
import timeit
import logging
import logging.handlers
import argparse

arg_parse = argparse.ArgumentParser()
arg_parse.add_argument("-v", "--voc_size", default=100, type=int,
                       help="Number of distinct tokens in the vocabulary.")
arg_parse.add_argument("-d", "--dir",
                       help="Directory for storing model output.")
arg_parse.add_argument("-f", "--filename",
                       help="Name of the log file to use.")
arg_parse.add_argument("-e", "--epochs", default=10, type=int,
                       help="Number of epochs for which to train the RNN.")
arg_parse.add_argument("-m", "--max", default=20, type=int,
                       help="The maximum number of examples to train on.")
arg_parse.add_argument("-t", "--test", action="store_true",
                       help="Treat run as test, do not save models")
arg_parse.add_argument("-l", "--learning_rate", default=0.005, type=float,
                       help="The initial learning rate used for training.")
args = arg_parse.parse_args()

if args.dir == None:
    sentenceDir = "./grurnn/" + time.strftime("%d%m%y%H") + "/sentences/"
    modelDir = "./grurnn/" + time.strftime("%d%m%y%H") + "/models/"
    logDir = "./grurnn/" + time.strftime("%d%m%y%H") + "/logs"
else:
    sentenceDir = args.dir + time.strftime("%d%m%y%H") + "/sentences/"
    modelDir = args.dir + time.strftime("%d%m%y%H") + "/models/"
    logDir = args.dir + time.strftime("%d%m%y%H") + "/logs/"

if args.filename == None:
    logFile = "vanilla.log"
else:
    logFile = args.filename

RNN.createDir(sentenceDir)
RNN.createDir(modelDir)
RNN.createDir(logDir)

testlog = logging.getLogger("TEST")
testlog.setLevel(logging.INFO)

handler = logging.handlers.RotatingFileHandler(
    filename=logDir+"/"+logFile,
    maxBytes=1024*512,
    backupCount=5
)

formatter = logging.Formatter(
    "%(asctime)s-%(name)s-%(levelname)s-%(message)s"
)

handler.setFormatter(formatter)

testlog.addHandler(handler)
testlog.info("Running a new VanillaRNN with logging")

model = RNN.GruRNN(voc_size=args.voc_size, hid_size=50, trunc=1000)
model.load_data("datasets/test.pkl")

model.train_rnn(
    epochs=args.epochs,
    patience=20 if args.max == None else args.max,
    path=modelDir+"/reladred",
    max=args.max,
    testing=False,
    learning_rate=args.learning_rate
)

testlog.info("Generating sentences")

file = open(sentenceDir+"/sentences.txt", "w")

attempts = 0
sents = 0

while sents < 10:
    sentence = model.generate_sentence()
    file.write(" ".join(sentence) + "\n")
    sents += 1
    attempts += 1

file.close()

testlog.info("Generated %d sentences after %d attempts." %
             (sents, attempts))
