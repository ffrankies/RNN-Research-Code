#import theano
#theano.config.cxx = ""

import io
import sys
import os
import time
import timeit
import argparse
import logging
import logging.handlers
from threading import Thread, Lock

import vanillarnn as VanillaRnn
import grurnn as GruRnn
import adamrnn as AdamRnn

#
# Global variables
#

# The log used for logging
log = None

# Holds the threads currently running
threads = []

# Holds all the Neural Networks
networks = []

# Holds the names of all the Neural Networks
networkNames = []

# Holds the time taken to run gradient descent each Neural Network
gradDescentTimes = dict()

# Holds the time taken to train each Neural Network
trainingTimes = dict()

# Holds the time taken to generate sentences for each Neural Network
generateTimes = dict()

# Lock to make accessing networks thread safe
LOCK = Lock()

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

def setUpLogger(logDir=None, logFileName=None):
    """
    Sets up logging for the test suite.
    """
    if logDir == None:
        logDir = ".logs"
    if logFileName == None:
        logFileName = "testSuite.log"

    global log
    log = logging.getLogger("TEST")
    log.setLevel(logging.DEBUG)

    createDir(".logs")

    handler = logging.handlers.RotatingFileHandler(
        filename=".logs/testSuite.log",
        maxBytes=1024*5,
        backupCount=5
    )
    formatter = logging.Formatter(
        "%(asctime)s-%(name)s-%(levelname)s-%(message)s"
    )
    handler.setFormatter(formatter)
    log.addHandler(handler)
    log.info("Started a new Test Suite with logging")
# End setUpLogger()

def setUpRnns(vanillarnn=True, grurnn=True, adamrnn=True):
    """
    Sets up the requested RNNs.
    """
    logText = "Initialized requested networks:"

    if vanillarnn == True:
        networks.append(VanillaRnn.VanillaRNN())
        networks[-1].load_data()
        logText += " VanillaRNN"
        networkNames.append("VanillaRNN")

    if grurnn == True:
        networks.append(GruRnn.GruRNN())
        networks[-1].load_data()
        logText += " GRURNN"
        networkNames.append("GRU-RNN")

    if adamrnn == True:
        networks.append(AdamRnn.AdamRNN())
        networks[-1].load_data()
        logText += " AdamRNN"
        networkNames.append("Adam-RNN")

    log.info(logText + ".")
# End of setUpRnns()

def _testGradDescent(index):
    """
    Meant to be a private function, to be ran inside a thread.
    Tests the gradient descent of one Neural Network, using 20 example inputs.
    Training will run for a maxinum of 500 epochs, or until a loss of 0 is
    incurred.

    :type index:
    :param index: the index of the network in the Networks list
    """
    global GradDescentTimes
    log.info("Testing gradient descent of " + networkNames[index])
    LOCK.acquire()
    gradDescentTimes[networkNames[index]] = (-1, -1)
    LOCK.release()
    (loss, time) = networks[index].train_rnn(epochs=500, max=20, testing=True)
    LOCK.acquire()
    gradDescentTimes[networkNames[index]] = (loss, time)
    LOCK.release()
# End of _testGradDescent()

def testGradDescent():
    """
    Tests whether or not gradient descent works for every specified network,
    using 20 example inputs.
    Training will run for a maxinum of 500 epochs, or until a loss of 0 is
    incurred.
    """
    # Clear list of threads
    global threads
    threads = []

    for i in range(len(networks)):
        threads.append(
            Thread(
                name=networkNames[i],
                target=_testGradDescent,
                args=(i, )
            )
        )
        threads[-1].start()

    for i in range(len(threads)):
        threads[-1].join()

    log.info("Gradient Descent Testing Complete. Summary:")
    for i in gradDescentTimes.keys():
        logText = str(i) + " got a loss of " + str(gradDescentTimes[i][0])
        logText += " after training for " + str(gradDescentTimes[i][1])
        logText += " minutes."
        log.info(logText)
# End testGradDescent()

def _trainNetworks(index, epochs=10, savePath="models/"):
    """
    Private method for training a single network given a set number of epochs
    and a save path for the generated models.

    :type index: int
    :param index: the index of the network in the Networks list

    :type epochs: int
    :param epochs: the maximum number of epochs for which the models will be
                   trained. Defaults to 10.

    :type savePath: string
    :param savePath: the directory string referencing the directory in which
                     models will be stored.
    """
    global trainingTimes
    global time
    log.info("Training " + networkNames[index])
    LOCK.acquire()
    trainingTimes[networkNames[index]] = (-1, -1)
    LOCK.release()
    savePath += time.strftime("%d%m%y%H") + "/" + networkNames[index]
    createDir(savePath)
    (loss, time) = networks[index].train_rnn(epochs=epochs, path=savePath)
    LOCK.acquire()
    trainingTimes[networkNames[index]] = (loss, time)
    LOCK.release()
# End of _trainNetworks()

def trainNetworks(epochs=10, savePath="models/"):
    """
    Trains each requested RNN for a given number of epochs. Saves the models
    in a provided directory under a specified savePath. Each model will be
    saved in a subdirectory with the model's name.

    :type epochs: int
    :param epochs: the maximum number of epochs for which the models will be
                   trained. Defaults to 10.

    :type savePath: string
    :param savePath: the directory string referencing the directory in which
                     models will be stored.
    """
    # Clear list of threads
    global threads
    threads = []

    for i in range(len(networks)):
        threads.append(
            Thread(
                name=networkNames[i],
                target=_trainNetworks,
                args=(i, epochs, savePath, )
            )
        )
        threads[-1].start()

    for i in range(len(threads)):
        threads[-1].join()

    log.info("Training of Networks for %d epochs Complete. Summary:" % epochs)
    for i in trainingTimes.keys():
        logText = str(i) + " got a loss of " + str(trainingTimes[i][0])
        logText += " after training for " + str(trainingTimes[i][1])
        logText += " minutes."
        log.info(logText)
# End of trainNetworks

def _generateSentences(index, num=100, minLength=5, saveDir="sentences/"):
    """
    Private method that generates sentences from one RNN and saves them in
    a file under a specified directory.

    :type index: int
    :param index: the index of the network in the Networks list

    :type num: int
    :param num: the number of sentences to generate

    :type minLength: int
    :param minLength: the minimum number of sentences to generate

    :type saveDir: string
    :param saveDir: the directory in which to save the generated sentences
    """
    global generateTimes
    global time
    log.info("Generating sentences with " + networkNames[index])
    LOCK.acquire()
    generateTimes[networkNames[index]] = (-1, -1)
    LOCK.release()
    createDir(savePath + networkNames[index])
    savePath += networkNames[index] + "/" + time.strftime("%d%m%y%H") + ".txt"
    RNN = networks[index]

    start_time = timeit.default_timer()

    attempt = 1

    with open(savePath) as outFile:

        num_generated = 0
        while num_generated < num:
            sentence = RNN.generate_sentence()
            print("Attempt %d: %s" % (attempt, " ".join(sentence)))
            sys.stdout.flush()
            if len(sentence) > minLength:
                # print(" ".join(sentence))
                outFile.write(u' '.join(sentence).encode('utf-8') + "\n")
                sys.stdout.flush()
                num_generated += 1
            attempt += 1

        outFile.close()

    end_time = timeit.default_timer()
    timeTaken = (start_time - end_time) / 60

    LOCK.acquire()
    generateTimes[networkNames[index]] = (timeTaken, attempt)
    LOCK.release()
# End of _generateSentences()

def generateSentences(num=100, minLength=5, saveDir="sentences/"):
    """
    Generates sentences from all the requested networks and saves them in
    a file under a specified directory.

    :type num: int
    :param num: the number of sentences to generate

    :type minLength: int
    :param minLength: the minimum number of sentences to generate

    :type saveDir: string
    :param saveDir: the directory in which to save the generated sentences
    """
    # Clear list of threads
    global threads
    threads = []

    for i in range(len(networks)):
        threads.append(
            Thread(
                name=networkNames[i],
                target=_generateSentences,
                args=(i, num, minLength, saveDir, )
            )
        )
        threads[-1].start()

    for i in range(len(threads)):
        threads[-1].join()

    log.info("Generation of sentences Complete. Summary:")
    for i in generateTimes.keys():
        logText = str(i) + " generated " + str(num) + " valid sentences after "
        logText += str(generateTimes[i][0]) + " minutes and "
        logText += str(generateTimes[i][1]) + " attempts."
        log.info(logText)
# End generateSentences

def main():
    """
    Main method sets up logging, initiates the Networks, and runs all
    applicable tests.
    """
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-v", "--vanilla", action="store_true",
                            help="Run a test on the Vanilla RNN")
    arg_parser.add_argument("-g", "--gru", action="store_true",
                            help="Run a test on the GRU RNN")
    arg_parser.add_argument("-a", "--adam", action="store_true",
                            help="Run a test on the ADAM RNN")
    arg_parser.add_argument("-tg", "--testgrad", action="store_true",
                            help="Run the gradient descent test.")
    arg_parser.add_argument("-tr", "--train", action="store_true",
                            help="Train the network(s).")
    arg_parser.add_argument("-ge", "--generate", action="store_true",
                            help="Generate sentences from network(s).")
    arg_parser.add_argument("-l", "--logdir", 
                            help="The directory in which logs will be stored.")
    arg_parser.add_argument("-f", "--logfname",
                            help="The filename in which logs will be saved.")

    cmd_args = arg_parser.parse_args()

    setUpLogger(cmd_args.logdir, cmd_args.logfname)

    if cmd_args.vanilla or cmd_args.gru or cmd_args.adam: # Run specified RNNs
        setUpRnns(cmd_args.vanilla, cmd_args.gru, cmd_args.adam)
    else: # Run all models
        setUpRnns()

    if cmd_args.testgrad or cmd_args.train or cmd_args.generate:
        # Run specified tests
        if cmd_args.testgrad:
            testGradDescent()
        if cmd_args.train:
            trainNetworks(epochs=20)
        if cmd_args.generate:
            generateSentences()
    else: # Run all tests
        testGradDescent()
        trainNetworks(epochs=20)
        generateSentences()
# End of main()

if __name__ == "__main__":
    main()
