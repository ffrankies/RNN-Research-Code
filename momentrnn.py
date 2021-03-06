###############################################################################
# Source: RNN tutorial from www.wildml.com
#
# The following is a python3 implementation of a  Recurrent Neural Network with
# GRU units and using the Nesterov's momentum algorithm in place of simple
# stochastic gradient descent and the Adam algorithm for gradient descent,
# to predict the next word in a sentence. The nature of this network is that it
# can be used to generate sentences that will resemble the format of the
# training set. This particular version was meant to be used with reddit
# comment datasets, although it should in theory work fine with other
# sentence-based datasets, as well.
# The various datasets that this particular RNN will be used for will be
# kernels of the main dataset of reddit comments from May 2015 fourd on kaggle
# Dataset source: https://www.kaggle.com/reddit/reddit-comments-may-2015
#
# Author: Frank Derry Wanye
# Date: 04 Feb 2017
###############################################################################

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

class MomentRNN(object):
    """
    A Recurrent Neural Network with GRU units using the Nesterov's momentum
    algorithm in place of the simpler stochastic gradient descent, or the more
    complex Adam algorithm, for a balance between training time and optimized
    convergence. The algorithm should speed up the training of this network as
    well as make it slighlty more precise. The network essentially looks like
    this, with the hidden units being GRU units:
        output      output      output      ...     output
          |           |           |                   |
    ----hidden------hidden------hidden------...-----hidden----
          |           |           |                   |
        input       input       input       ...     input
    The wght in this network are shared between the horizontal layers.
    The input and output for each horizontal layer are in the form of a vector
    (list, in the python implementation) of size len(vocabulary). For the
    input, the vector is a one-hot vector - it is comprised of all zeros,
    except for the index that corresponds to the input word, where the value is
    1. The output vector will contain the probabilities for each possible word
    being the next word. The word chosen as the actual output will the word
    corresponding to the index with the highest probability.

    :author: Frank Wanye
    :date: 04 Feb 2017
    """

    def __init__(self, voc_size=8000, hid_size=100, trunc=4, model=None):
        """
        Initializes the Vanilla RNN with the provided vocabulary_size,
        hidden layer size, and bptt_truncate. Also initializes the functions
        used in this RNN.

        :type vocabulary_size: int
        :param vocabulary_size: the size of the RNN's vocabulary. Determines
                                the number of input and output neurons in the
                                network. Default: 8000

        :type hidden_size: int
        :param hidden_size: the number of hidden layer neurons in the network.
                            Default: 100

        :type bptt_truncate: int
        :param bptt_truncate: how far back back-propagation-through-time goes.
                              This is a crude method of reducing the effect
                              of vanishing/exploding gradients, as well as
                              reducing training time, since the network won't
                              have to go through every single horizontal layer
                              during training. NOTE: this causes the accuracy
                              of the network to decrease. Default: 4

        :type model: string
        :param model: the name of the saved model that contains all the RNN
                      info.
        """

        self.log = logging.getLogger("TEST.ADAM")
        self.log.setLevel(logging.INFO)

        self.unknown_token = "UNKNOWN_TOKEN"
        self.sentence_start_token = "SENTENCE_START"
        self.sentence_end_token = "SENTENCE_END"

        if model == None:
            self.log.info("Initializing RNN parameters and functions...")


            self.vocabulary_size = voc_size
            self.hidden_size = hid_size
            self.bptt_truncate = trunc

            # Instantiate the network wght
            # I feel like the first and third are switched for some reason...
            # but it's pretty consistent in the example code. Perhaps it's
            # backwards for a purpose
            # The wght going from input layer to hidden layer
            # (U, in tutorial)
            wght_ih = np.random.uniform(-np.sqrt(1./voc_size),
                                            np.sqrt(1./voc_size),
                                            (3, hid_size, voc_size))
            # The wght going from hidden layer to hidden layer
            # (W, in tutorial)
            wght_hh = np.random.uniform(-np.sqrt(1./voc_size),
                                            np.sqrt(1./voc_size),
                                            (3, hid_size, hid_size))
            # The wght going from hidden layer to output layer
            # (V, in tutorial)
            wght_ho = np.random.uniform(-np.sqrt(1./voc_size),
                                            np.sqrt(1./voc_size),
                                            (voc_size, hid_size))
            # The bias for the hidden units
            bias = np.zeros((3, hid_size))
            # The bias for the output units
            out_bias = np.zeros(voc_size)

            #
            # Initialize wght
            #
            self.wght_ih = theano.shared(
                name='wght_ih',
                value=wght_ih.astype(theano.config.floatX))

            self.wght_hh = theano.shared(
                name='wght_hh',
                value=wght_hh.astype(theano.config.floatX))

            self.wght_ho = theano.shared(
                name='wght_ho',
                value=wght_ho.astype(theano.config.floatX))

            self.bias = theano.shared(
                name='bias',
                value=bias.astype(theano.config.floatX))

            self.out_bias = theano.shared(
                name='out_bias',
                value=out_bias.astype(theano.config.floatX))

            #
            # Initialize Adam parameters. Adam works through 'm' vectors,
            # 2 for each set of wght.
            #
            self.m_w_ih = theano.shared(
                name='m_w_ih',
                value=np.zeros((2, 3, hid_size, voc_size)).astype(
                      theano.config.floatX))

            self.m_w_hh = theano.shared(
                name='m_w_hh',
                value=np.zeros((2, 3, hid_size, hid_size)).astype(
                      theano.config.floatX))

            self.m_w_ho = theano.shared(
                name='m_w_ho',
                value=np.zeros((2, voc_size, hid_size)).astype(
                      theano.config.floatX))

            self.m_bias = theano.shared(
                name='m_bias',
                value=np.zeros((2, 3, hid_size)).astype(theano.config.floatX))

            self.m_out_bias = theano.shared(
                name='m_out_bias',
                value=np.zeros((2, voc_size)).astype(
                      theano.config.floatX))

            decay = (0.9, 0.999)
            eps = 0.00000001 # Prevents division by 0

            self.decay = theano.shared(
                name='decay',
                value=np.asarray(decay).astype(theano.config.floatX))

            self.eps = theano.shared(
                name='eps',
                value=np.cast["float32"](eps))

            #
            # Initialize empty vectors for vocabulary
            #
            self.vocabulary = []
            self.word_to_index = {}
            self.index_to_word = []
        else:
            self.log.info("Loading model parameters from saved model...")


            with open(model, "rb") as modelFile:
                params = cPickle.load(modelFile)

                self.vocabulary_size = params[0]
                self.hidden_size = params[1]
                self.bptt_truncate = params[2]

                self.wght_ih = params[3]
                self.wght_hh = params[4]
                self.wght_ho = params[5]

                self.vocabulary = params[6]
                if not self.vocabulary[-1] == self.unknown_token:
                    self.log.info("Appending unknown token")
                    self.vocabulary[-1] = self.unknown_token
                self.index_to_word = params[7]
                self.word_to_index = params[8]

                self.bias = params[9]
                self.out_bias = params[10]

                self.m_w_ih = params[11]
                self.m_w_hh = params[12]
                self.m_w_ho = params[13]
                self.m_bias = params[14]
                self.m_out_bias = params[15]
                self.decay = params[16]
                self.eps = params[17]
        # End of if statement

        # Symbolic representation of one input sentence
        input = T.ivector('sentence')

        # Symbolic representation of the one output sentence
        output = T.ivector('sentence')

        def forward_propagate(word, previous_state):
            """
            Vertically propagates one of the words.

            :type word: int
            :param word: the index of the current input word

            :type previous_state: T.dvector()
            :param word: the output of the hidden layer from the previous
                         horizontal layer
            """
            # GRU layer
            update_gate = T.nnet.hard_sigmoid(
                self.wght_ih[0][:, word] +
                self.wght_hh[0].dot(previous_state) +
                self.bias[0]
            )

            reset_gate = T.nnet.hard_sigmoid(
                self.wght_ih[1][:, word] +
                self.wght_hh[1].dot(previous_state) +
                self.bias[1]
            )

            hypothesis = T.tanh(
                self.wght_ih[2][:, word] +
                self.wght_hh[2].dot(previous_state * reset_gate) +
                self.bias[2]
            )

            temp = T.ones_like(update_gate) - update_gate
            current_state = temp * hypothesis + update_gate * previous_state

            current_output = T.nnet.softmax(
                self.wght_ho.dot(current_state) + self.out_bias
            )[0]

            # Not sure why current_output[0] and not just current_output...
            return [current_output, current_state]

        #######################################################################
        # Symbolically represents going through each input sentence word and
        # then calculating the state of the hidden layer and output word for
        # each word. The forward_propagate function is the one used to
        # generate the output word and hidden layer state.
        #######################################################################
        self.theano = {}

        [out, state], updates = theano.scan(
            forward_propagate,
            sequences=input,
            truncate_gradient=self.bptt_truncate,
            outputs_info=[None,
                          dict(initial=T.zeros(self.hidden_size))]
        )

        # Predicts the output words for each word in the sentence
        prediction = T.argmax(out, axis=1)

        # Calculates the output error between the predicted output and the
        # actual output
        out_error = T.sum(T.nnet.categorical_crossentropy(out, output))

        # Symbolically represents gradient calculations for gradient descent
        d_wght_ih = T.grad(out_error, self.wght_ih)
        d_wght_hh = T.grad(out_error, self.wght_hh)
        d_wght_ho = T.grad(out_error, self.wght_ho)
        d_bias = T.grad(out_error, self.bias)
        d_out_bias = T.grad(out_error, self.out_bias)

        # Symbolic theano functions
        self.forward_propagate = theano.function([input], out)
        self.predict = theano.function([input], prediction)
        self.calculate_error = theano.function([input, output], out_error)
        self.bptt = theano.function([input, output],
            [d_wght_ih, d_wght_hh, d_wght_ho, d_bias, d_out_bias])

        self.learning_rate = 0.005

        # Chaining Adam subtensors for updates
        new_m_w_ih = self.m_w_ih
        new_m_w_ih = T.set_subtensor(new_m_w_ih[0], self.decay[0] *
                     new_m_w_ih[0] + (1 - self.decay[0]) * d_wght_ih)
        new_m_w_ih = T.set_subtensor(new_m_w_ih[1], self.decay[1] *
                     new_m_w_ih[1] + (1 - self.decay[1]) * (d_wght_ih ** 2))

        new_m_w_hh = self.m_w_hh
        new_m_w_hh = T.set_subtensor(new_m_w_hh[0], self.decay[0] *
                     new_m_w_hh[0] + (1 - self.decay[0]) * d_wght_hh)
        new_m_w_hh = T.set_subtensor(new_m_w_hh[1], self.decay[1] *
                     new_m_w_hh[1] + (1 - self.decay[1]) * (d_wght_hh ** 2))

        new_m_w_ho = self.m_w_ho
        new_m_w_ho = T.set_subtensor(new_m_w_ho[0], self.decay[0] *
                     new_m_w_ho[0] + (1 - self.decay[0]) * d_wght_ho)
        new_m_w_ho = T.set_subtensor(new_m_w_ho[1], self.decay[1] *
                     new_m_w_ho[1] + (1 - self.decay[1]) * (d_wght_ho ** 2))

        new_m_bias = self.m_bias
        new_m_bias = T.set_subtensor(new_m_bias[0], self.decay[0] *
                     new_m_bias[0] + (1 - self.decay[0]) * d_bias)
        new_m_bias = T.set_subtensor(new_m_bias[1], self.decay[1] *
                     new_m_bias[1] + (1 - self.decay[1]) * (d_bias ** 2))

        new_m_out_bias = self.m_out_bias
        new_m_out_bias = T.set_subtensor(new_m_out_bias[0], self.decay[0] *
                         new_m_out_bias[0] + (1 - self.decay[0]) * d_out_bias)
        new_m_out_bias = T.set_subtensor(new_m_out_bias[1], self.decay[1] *
                         new_m_out_bias[1] + (1 - self.decay[1]) *
                         (d_out_bias ** 2))

        print("Shape of (np.sqrt(self.m_w_ih[1]) + self.eps)")
        print((np.sqrt(self.m_w_ih[1]) + self.eps).type)
        print("Shape of m_w_ih[0] / (np.sqrt(self.m_w_ih[1]) + self.eps)")
        print((self.m_w_ih[1] / (np.sqrt(self.m_w_ih[1]) + self.eps)).type)
        print("Shape of update")
        print((self.wght_ih - self.learning_rate * self.m_w_ih[0]
              / (np.sqrt(self.m_w_ih[1]) + self.eps)).type)

        # The list of updates needed for Adam SGD to work
        adam_updates = [
            (self.m_w_ih, new_m_w_ih),
            (self.m_w_hh, new_m_w_hh),
            (self.m_w_ho, new_m_w_ho),
            (self.m_bias, new_m_bias),
            (self.m_out_bias, new_m_out_bias),
            (self.wght_ih, self.wght_ih - self.learning_rate *
             self.m_w_ih[0] / (np.sqrt(self.m_w_ih[1]) + self.eps)),
            (self.wght_hh, self.wght_hh - self.learning_rate *
             self.m_w_hh[0] / (np.sqrt(self.m_w_hh[1]) + self.eps)),
            (self.wght_ho, self.wght_ho - self.learning_rate *
             self.m_w_ho[0] / (np.sqrt(self.m_w_ho[1]) + self.eps)),
            (self.bias, self.bias - self.learning_rate *
             self.m_bias[0] / (np.sqrt(self.m_bias[1]) + self.eps)),
            (self.out_bias, self.out_bias - self.learning_rate *
             self.m_out_bias[0] / (np.sqrt(self.m_out_bias[1]) +
             self.eps))
        ]

        # Adam SGD step function
        learning_rate = T.scalar('learning_rate')
        self.sgd_step = theano.function(
            [input, output], [],
            updates=adam_updates
        )

        self.x_train = None
        self.y_train = None
    # End of __init__()

    def calculate_total_loss(self, train_x, train_y):
        """
        Calculates the sum of the losses for a given epoch (sums up the losses
        for each sentence in train_x).

        :type train_x: T.imatrix()
        :param train_x: the training examples (list of tokenized and indexed
                        sentences, starting from SENTENCE_START and not
                        including SENTENCE_END)

        :type train_y: T.imatrix()
        :param train_y: the training solutions (list of tokenized and indexed
                        sentences, not including SENTNECE_START and going to
                        SENTENCE_END)
        """
        return np.sum([self.calculate_error(x, y)
                       for x, y in zip(train_x, train_y)])
    # End of calculate_total_loss()

    def calculate_loss(self, train_x, train_y):
        """
        Calculates the average loss for a given epoch (the average of the
        output of calculate_total_loss())

        :type train_x: T.imatrix()
        :param train_x: the training examples (list of tokenized and indexed
                        sentences, starting from SENTENCE_START and not
                        including SENTENCE_END)

        :type train_y: T.imatrix()
        :param train_y: the training solutions (list of tokenized and indexed
                        sentences, not including SENTNECE_START and going to
                        SENTENCE_END)
        """
        self.log.info("Calculating average categorical crossentropy loss...")

        num_words = np.sum([len(y) for y in train_y])
        return self.calculate_total_loss(train_x, train_y)/float(num_words)
    # End of calculate_loss()

    def load_data(self, filePath="reladred.pkl"):
        """
        Loads previously saved data from a pickled file.
        The number of vocabulary words must match self.vocabulary_size - 1 or
        else the dataset will not work.

        :type filePath: string
        :param filePath: the path to the file containing the dataset.
        """
        self.log.info("Loading the dataset from %s" % filePath)

        file = open(filePath, "rb")
        vocabulary, index_to_word, word_to_index, x_train, y_train = cPickle.load(file)

        self.log.info("Dataset contains %d words" % len(vocabulary))


        self.vocabulary = vocabulary
        self.index_to_word = index_to_word
        self.word_to_index = word_to_index
        self.x_train = x_train
        self.y_train = y_train
    # End of calculate_loss()

    def train_rnn(self, learning_rate=0.005, epochs=1, patience=10000,
                  path=None, max=None, testing=False):
        """
        Trains the RNN using stochastic gradient descent. Saves model
        parameters after every epoch.

        :type learning_rate: float
        :param learning_rate: multiplier for how much the wght of the
                              network get adjusted during gradient descent

        :type epochs: int
        :param epochs: the number of epochs (iterations over entire dataset)
                       for which the network will be trained.

        :type patience: int
        :param patience: the number of examples after which the loss should be
                         measured.

        :type path: string
        :param path: the path to the file in which the models should be stored.
                     The epoch number and .pkl extension will be added to the
                     path automatically.

        :type max: int
        :param max: the maximum number of examples it from the training set
                    used in the training
        """
        if self.x_train == None or self.y_train == None:
            self.log.info("Need to load data before training the rnn")

            return

        # Keep track of losses so that they can be plotted
        start_time = timeit.default_timer()

        self.learning_rate = learning_rate

        losses = []
        examples_seen = 0

        # Evaluate loss before training
        self.log.info("Evaluating loss before training.")

        if max == None:
            loss = self.calculate_loss(self.x_train, self.y_train)
        else:
            loss = self.calculate_loss(self.x_train[:max],
                                       self.y_train[:max])

        losses.append((examples_seen, loss))

        self.log.info("RNN incurred a loss of %f before training" % loss)

        for e in range(epochs):
            epoch = e + 1
            self.log.info("Training the model: epoch %d" % epoch)

            # Train separately for each training example (no need for
            # minibatches)
            if max == None:
                for example in range(len(self.y_train)):
                    self.sgd_step(self.x_train[example], self.y_train[example])
                    examples_seen += 1
                    if examples_seen % patience == 0:
                        self.log.info("Evaluated %d examples" % examples_seen)

            else:
                for example in range(len(self.y_train[:max])):
                    self.sgd_step(self.x_train[example], self.y_train[example])
                    examples_seen += 1
                    if examples_seen % patience == 0:
                        self.log.info("Evaluated %d examples" % examples_seen)
            # End of training for epoch

            # Evaluate loss after every epoch
            self.log.info("Evaluating loss: epoch %d" % epoch)

            if max == None:
                loss = self.calculate_loss(self.x_train, self.y_train)
            else:
                loss = self.calculate_loss(self.x_train[:max],
                                           self.y_train[:max])
            losses.append((examples_seen, loss))
            self.log.info("RNN incurred a loss of %f after %d epochs" %
                  (loss, epoch))
            # End of loss evaluation

            # Adjust learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                self.learning_rate = self.learning_rate * 0.5
                self.log.info("Setting learning rate to %f" % learning_rate)


            # End training if incurred a loss of 0
            if losses[-1][1] == 0:
                end_time = timeit.default_timer()

                self.log.info(
                    ("Finished training the rnn for %d epochs, with a final" +
                     " loss of %f. Training took %.2f m") %
                    (epochs, losses[-1][1], (end_time - start_time) / 60)
                )
                return (0, (end_time - start_time) / 60)

            # Saving model parameters
            if testing == False:
                params = (
                    self.vocabulary_size, self.hidden_size, self.bptt_truncate,
                    self.wght_ih, self.wght_hh, self.wght_ho,
                    self.vocabulary, self.index_to_word, self.word_to_index,
                    self.bias, self.out_bias, self.m_w_ih, self.m_w_hh,
                    self.m_w_ho, self.m_bias, self.m_out_bias, self.decay,
                    self.eps
                )

                if path == None:
                    modelPath = "models/adamreladred" + str(epoch) + ".pkl"
                    with open(modelPath, "wb") as file:
                        cPickle.dump(params, file, protocol=2)
                else:
                    with open(path + str(epoch) + ".pkl", "wb") as file:
                        cPickle.dump(params, file, protocol=2)
        # End of training

        end_time = timeit.default_timer()

        self.log.info(
            ("Finished training the rnn for %d epochs, with a final loss of %f."
             + " Training took %.2f m") %
            (epochs, losses[-1][1], (end_time - start_time) / 60)
        )
        return (losses[-1][1], (end_time - start_time) / 60)
    # End of train_rnn()

    def generate_sentence(self):
        """
        Generates one sentence based on current model parameters. Model needs
        to be loaded or trained before this step in order to produce any
        results.

        :return type: list of strings
        :return param: a generated sentence, with each word being an item in
                       the array.
        """
        if self.word_to_index == None:
            self.log.info("Need to load a model or data before this step.")

            return []
        # Start sentence with the start token
        sentence = [self.word_to_index[self.sentence_start_token]]
        # Predict next word until end token is received
        while not sentence[-1] == self.word_to_index[self.sentence_end_token]:
            next_word_probs = self.forward_propagate(sentence)
            sampled_word = self.word_to_index[self.unknown_token]
            # We don't want the unknown token to appear in the sentence
            while sampled_word == self.word_to_index[self.unknown_token]:
                samples = np.random.multinomial(1, next_word_probs[-1])
                sampled_word = np.argmax(samples)
            sentence.append(sampled_word)
        sentence_str = [self.index_to_word[word] for word in sentence[1:-1]]
        return sentence_str
    # End of generate_sentence()

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

if __name__ == "__main__":
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("-d", "--dir", default="./adamrnn",
                           help="Directory for storing logs.")
    arg_parse.add_argument("-f", "--filename", default="adamrnn.log",
                           help="Name of the log file to use.")
    arg_parse.add_argument("-e", "--epochs", default=10, type=int,
                           help="Number of epochs for which to train the RNN.")
    arg_parse.add_argument("-m", "--max", default=None,
                           help="The maximum number of examples to train on.")
    arg_parse.add_argument("-p", "--patience", default=100000, type=int,
                           help="Number of examples to train before evaluating"
                                + " loss.")
    arg_parse.add_argument("-t", "--test", action="store_true",
                           help="Treat run as test, do not save models")
    arg_parse.add_argument("-o", "--model", default=None,
                           help="Previously trained model to load on init.")
    arg_parse.add_argument("-l", "--learn_rate", default=0.005, type=float,
                           help="The learning rate to be used in training.")
    args = arg_parse.parse_args()

    argsdir = args.dir + "/" + time.strftime("%d%m%y%H") + "/";
    sentenceDir = argsdir + "sentences/"
    modelDir = argsdir + "models/"
    logDir = argsdir + "logs/"
    logFile = args.filename

    createDir(sentenceDir)
    createDir(modelDir)
    createDir(logDir)

    testlog = logging.getLogger("TEST")
    testlog.setLevel(logging.INFO)

    handler = logging.handlers.RotatingFileHandler(
        filename=logDir+logFile,
        maxBytes=1024*512,
        backupCount=5
    )

    formatter = logging.Formatter(
        "%(asctime)s-%(name)s-%(levelname)s-%(message)s"
    )

    handler.setFormatter(formatter)

    testlog.addHandler(handler)
    testlog.info("Running a new Adam-RNN with logging")

    RNN = AdamRNN(model=args.model)
    RNN.load_data()
    #loss = RNN.calculate_loss(RNN.x_train, RNN.y_train)
    #self.log.info(loss)
    RNN.train_rnn(
        epochs=args.epochs,
        patience=args.patience,
        path=modelDir+"reladred",
        max=args.max,
        testing=args.test
    )

    if args.test:
        testlog.info("Finished running test.")
        sys.exit(0)

    testlog.info("Generating sentences")

    file = open(sentenceDir+"sentences.txt", "w")

    attempts = 0
    sents = 0

    while sents < 100:
        sentence = RNN.generate_sentence()
        if len(sentence) >= 5:
            file.write(" ".join(sentence) + "\n")
            sents += 1
        attempts += 1

    file.close()

    testlog.info("Generated %d sentences after %d attempts." %
                 (sents, attempts))
