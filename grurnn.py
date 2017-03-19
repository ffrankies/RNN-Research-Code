###############################################################################
# Source: RNN tutorial from www.wildml.com
#
# The following is a python3 implementation of a  Recurrent Neural Network with
# GRU units to predict the next word in a sentence. The nature of this
# network is that it can be used to generate sentences that will resemble
# the format of the training set. This particular version was meant to be used
# with reddit comment datasets, although it should in theory work fine with
# other sentence-based datasets, as well.
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
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt # For plotting

class GruRNN(object):
    """
    A Recurrent Neural Network that GRU units (neurons) in the hidden layer for
    combating vanishing and exploding gradients. The GRU is similar to the
    LSTM, but involves less computations, and is therefore more efficient. The
    network essentially looks like this, with the hidden units being GRU units:
        output      output      output      ...     output
          |           |           |                   |
    ----hidden------hidden------hidden------...-----hidden----
          |           |           |                   |
        input       input       input       ...     input
    The weights in this network are shared between the horizontal layers.
    The input and output for each horizontal layer are in the form of a vector
    (list, in the python implementation) of size len(vocabulary). For the
    input, the vector is a one-hot vector - it is comprised of all zeros,
    except for the index that corresponds to the input word, where the value is
    1. The output vector will contain the probabilities for each possible word
    being the next word. The word chosen as the actual output will the word
    corresponding to the index with the highest probability.

    :author: Frank Wanye
    :date: 19 Mar 2017
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

        self.log = logging.getLogger("TEST.GRU")
        self.log.setLevel(logging.INFO)

        self.unknown_token = "UNKNOWN_TOKEN"
        self.sentence_start_token = "SENTENCE_START"
        self.sentence_end_token = "SENTENCE_END"
        self.paragraph_start = "PARAGRAPH_START"
        self.paragraph_end = "PARAGRAPH_END"
        self.story_start = "STORY_START"
        self.story_end = "STORY_END"

        if model is None:
            self.log.info("Initializing RNN parameters and functions...")

            self.vocabulary_size = voc_size
            self.hidden_size = hid_size
            self.bptt_truncate = trunc

            # Instantiate the network weights
            # I feel like the first and third are switched for some reason...
            # but it's pretty consistent in the example code. Perhaps it's
            # backwards for a purpose
            # The weights going from input layer to hidden layer
            # (U, in tutorial)
            weights_ih = np.random.uniform(-np.sqrt(1./voc_size),
                                            np.sqrt(1./voc_size),
                                            (3, hid_size, voc_size))
            # The weights going from hidden layer to hidden layer
            # (W, in tutorial)
            weights_hh = np.random.uniform(-np.sqrt(1./voc_size),
                                            np.sqrt(1./voc_size),
                                            (3, hid_size, hid_size))
            # The weights going from hidden layer to output layer
            # (V, in tutorial)
            weights_ho = np.random.uniform(-np.sqrt(1./voc_size),
                                            np.sqrt(1./voc_size),
                                            (voc_size, hid_size))
            # The bias for the hidden units
            bias = np.zeros((3, hid_size))
            # The bias for the output units
            out_bias = np.zeros(voc_size)

            self.weights_ih = theano.shared(
                name='weights_ih',
                value=weights_ih.astype(theano.config.floatX))

            self.weights_hh = theano.shared(
                name='weights_hh',
                value=weights_hh.astype(theano.config.floatX))

            self.weights_ho = theano.shared(
                name='weights_ho',
                value=weights_ho.astype(theano.config.floatX))

            self.bias = theano.shared(
                name='bias',
                value=bias.astype(theano.config.floatX))

            self.out_bias = theano.shared(
                name='out_bias',
                value=out_bias.astype(theano.config.floatX))

            self.vocabulary = []
            self.word_to_index = {}
            self.index_to_word = []
        else:
            self.load_parameters(model)
        # End of if statement

        self.log.info("Network parameters: \n"
                      "Vocabulary size: %d\n"
                      "Hidden layer size: %d\n"
                      "Bptt truncate: %d\n" %
                      (self.vocabulary_size,
                       self.hidden_size,
                       self.bptt_truncate))

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
                self.weights_ih[0][:, word] +
                self.weights_hh[0].dot(previous_state) +
                self.bias[0]
            )

            reset_gate = T.nnet.hard_sigmoid(
                self.weights_ih[1][:, word] +
                self.weights_hh[1].dot(previous_state) +
                self.bias[1]
            )

            hypothesis = T.tanh(
                self.weights_ih[2][:, word] +
                self.weights_hh[2].dot(previous_state * reset_gate) +
                self.bias[2]
            )

            temp = T.ones_like(update_gate) - update_gate
            current_state = temp * hypothesis + update_gate * previous_state

            current_output = T.nnet.softmax(
                self.weights_ho.dot(current_state) + self.out_bias
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
        # Categorical crossentropy formula according to deeplearning.net:
        # -Sum(p(x)log(q(x))), where p(x) is the true distribution and q(x) is
        # the calculated distribution
        out_error = T.sum(T.nnet.categorical_crossentropy(
            out + T.mean(out)/1000, output))
        o_e = T.sum(
            -T.sum(
                T.extra_ops.to_one_hot(output, self.vocabulary_size)*T.log(out + T.mean(out)/1000)
            )
        )

        # Symbolically represents gradient calculations for gradient descent
        d_weights_ih = T.grad(out_error, self.weights_ih)
        d_weights_hh = T.grad(out_error, self.weights_hh)
        d_weights_ho = T.grad(out_error, self.weights_ho)
        d_bias = T.grad(out_error, self.bias)
        d_out_bias = T.grad(out_error, self.out_bias)

        # Symbolic theano functions
        self.forward_propagate = theano.function([input], out)
        self.predict = theano.function([input], prediction)
        self.calculate_error = theano.function([input, output], out_error)
        self.bptt = theano.function([input, output],
            [d_weights_ih, d_weights_hh, d_weights_ho, d_bias])

        # Stochastic Gradient Descent step
        learning_rate = T.scalar('learning_rate')

        ih_update = self.weights_ih - learning_rate * d_weights_ih
        hh_update = self.weights_hh - learning_rate * d_weights_hh
        ho_update = self.weights_ho - learning_rate * d_weights_ho
        bias_update = self.bias - learning_rate * d_bias
        out_bias_update = self.out_bias - learning_rate * d_out_bias

        self.sgd_step = theano.function(
            [input, output, learning_rate], [],
            updates=[(self.weights_ih, ih_update),
                     (self.weights_hh, hh_update),
                     (self.weights_ho, ho_update),
                     (self.bias, bias_update),
                     (self.out_bias, out_bias_update)]
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

    def load_parameters(self, model=None):
        """
        Loads the network parameters from a previously pickled file using the
        cPickle library.

        :type path: string
        :param path: the path to the pickled file containing the network
                     parameters.
        """
        self.log.info("Loading model parameters from saved model...")

        if model is None or model == "":
            self.log.error("No model provided.")
            sys.exit(1)

        with open(model, "rb") as modelFile:
            params = cPickle.load(modelFile)

            self.vocabulary_size = params[0]
            self.hidden_size = params[1]
            self.bptt_truncate = params[2]

            weights_ih = params[3]
            weights_hh = params[4]
            weights_ho = params[5]

            self.vocabulary = params[6]
            if not self.vocabulary[-1] == self.unknown_token:
                self.log.info("Appending unknown token")
                self.vocabulary[-1] = self.unknown_token
            self.index_to_word = params[7]
            self.word_to_index = params[8]

            bias = params[9]
            out_bias = params[10]

            self.weights_ih = theano.shared(
                name='weights_ih',
                value=weights_ih.astype(theano.config.floatX))

            self.weights_hh = theano.shared(
                name='weights_hh',
                value=weights_hh.astype(theano.config.floatX))

            self.weights_ho = theano.shared(
                name='weights_ho',
                value=weights_ho.astype(theano.config.floatX))

            self.bias = theano.shared(
                name='bias',
                value=bias.astype(theano.config.floatX))

            self.out_bias = theano.shared(
                name='out_bias',
                value=out_bias.astype(theano.config.floatX))
    # End of load_parameters()

    def save_parameters(self, path=None, epoch=0):
        """
        Saves the network parameters using the cPickle library.

        :type path: string
        :param path: the path to which the parameters will be saved.
        """
        params = (
            self.vocabulary_size,
            self.hidden_size,
            self.bptt_truncate,
            self.weights_ih.get_value(),
            self.weights_hh.get_value(),
            self.weights_ho.get_value(),
            self.vocabulary,
            self.index_to_word,
            self.word_to_index,
            self.bias.get_value(),
            self.out_bias.get_value()
        )

        if path is None:
            modelPath = "model.pkl"
            with open(modelPath, "wb") as file:
                cPickle.dump(params, file, protocol=2)
        else:
            with open(path + ".pkl", "wb") as file:
                cPickle.dump(params, file, protocol=2)
    # End of save_parameters()

    def train_rnn(self, learning_rate=0.005, epochs=1, patience=10000,
                  path=None, max=None, testing=False, anneal=0.000001):
        """
        Trains the RNN using stochastic gradient descent. Saves model
        parameters after every epoch.

        :type learning_rate: float
        :param learning_rate: multiplier for how much the weights of the
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
                    used in the training.

        :type testing: bool
        :param testing: if this is set to true, the models will not be saved
                        off.

        :type anneal: float
        :param anneal: the minimum value to which the learning rate can be
                       annealed.
        """
        if self.x_train is None or self.y_train is None:
            self.log.info("Need to load data before training the rnn")

            return

        # Keep track of losses so that they can be plotted
        start_time = timeit.default_timer()

        losses = []
        examples_seen = 0

        # Evaluate loss before training
        self.log.info("Evaluating loss before training.")

        if max is None:
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
            if max is None:
                for example in range(len(self.y_train)):
                    self.sgd_step(self.x_train[example], self.y_train[example],
                                  learning_rate)
                    examples_seen += 1
                    if examples_seen % patience == 0:
                        self.log.info("Evaluated %d examples" % examples_seen)

            else:
                for example in range(len(self.y_train[:max])):
                    self.sgd_step(self.x_train[example], self.y_train[example],
                                  learning_rate)
                    examples_seen += 1
                    if examples_seen % patience == 0:
                        self.log.info("Evaluated %d examples" % examples_seen)
            # End of training for epoch

            #Preventing zeros in weights
            #Retrieve numpy arrays of each network parameter
            local_w_ih = self.weights_ih.get_value()
            local_w_hh = self.weights_hh.get_value()
            local_w_ho = self.weights_ho.get_value()
            local_bias = self.bias.get_value()
            local_out_bias = self.out_bias.get_value()

            out = self.forward_propagate(self.x_train[3])
            print("%f - %f, %f" % (np.max(out), np.min(out), np.sum(out)))

            print("Max weight: %f, min weight: %f" %
                (np.max([
                    np.max(local_w_ih),
                    np.max(local_w_hh),
                    np.max(local_w_ho),
                    np.max(local_bias),
                    np.max(local_out_bias)]),
                 np.min([
                    np.min(local_w_ih),
                    np.min(local_w_hh),
                    np.min(local_w_ho),
                    np.min(local_bias),
                    np.min(local_out_bias)])))

            print("Mean out: %f" % np.mean(out))

            if np.isnan(local_bias).any() or np.isnan(local_out_bias).any() or np.isnan(local_w_ih).any() or np.isnan(local_w_hh).any() or np.isnan(local_w_ho).any():
                print("Found a nan inside the weights.")

            if np.isinf(local_bias).any() or np.isinf(local_out_bias).any() or np.isinf(local_w_ih).any() or np.isinf(local_w_hh).any() or np.isinf(local_w_ho).any():
                print("Found an infinity inside the weights.")

            # Alternative - use theano.tensor.clip and switch to switch zeros
            # for small numbers, and clip numbers to a given max and min value

            # Evaluate loss after every epoch
            self.log.info("Evaluating loss: epoch %d" % epoch)

            if max is None:
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
                if learning_rate > anneal:
                    learning_rate = learning_rate * 0.5
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
                self.save_parameters(path, epoch)
        # End of training

        end_time = timeit.default_timer()

        self.log.info(
            ("Finished training the rnn for %d epochs, with a final loss of %f."
             + " Training took %.2f m") %
            (epochs, losses[-1][1], (end_time - start_time) / 60)
        )

        # Plot a graph of loss against epochs, save graph
        self.log.info("Plotting a graph of loss vs iteration")
        plot_iterations = []
        plot_losses = []
        for i in range(len(losses)):
            plot_iterations.append(i)
            plot_losses.append(losses[i][1])
        plt.plot(plot_iterations, plot_losses)
        if path is None:
            modelPath = "loss_plot.png"
            plt.savefig(modelPath)
        else:
            modelPath = path + "loss_plot.png"
            plt.savefig(modelPath)

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
        if self.word_to_index is None:
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

    def generate_paragraph(self):
        """
        Generates one paragraph based on current model parameters. Model needs
        to be loaded or trained before this step in order to produce any
        results.

        :return type: list of strings
        :return param: a generated paragraph, with each word being an item in
                       the array.
        """
        if self.word_to_index is None:
            self.log.info("Need to load a model or data before this step.")
            return []

        # Start paragraph with the start token
        paragraph = [self.word_to_index[self.paragraph_start]]
        # Predict next word until end token is received
        while not paragraph[-1] == self.word_to_index[self.paragraph_end]:
            next_word_probs = self.forward_propagate(paragraph)
            sampled_word = self.word_to_index[self.unknown_token]
            # We don't want the unknown token to appear in the paragraph
            while sampled_word == self.word_to_index[self.unknown_token]:
                samples = np.random.multinomial(1, next_word_probs[-1])
                sampled_word = np.argmax(samples)
            paragraph.append(sampled_word)
        paragraph_str = [self.index_to_word[word] for word in paragraph[1:-1]]
        return paragraph_str
    # End of generate_paragraph()

    def generate_story(self):
        """
        Generates one story based on current model parameters. Model needs
        to be loaded or trained before this step in order to produce any
        results.

        :return type: list of strings
        :return param: a generated story, with each word being an item in
                       the array.
        """
        if self.word_to_index is None:
            self.log.info("Need to load a model or data before this step.")
            return []

        num_predictions = 0
        num_unknowns = 0

        # Start story with the start token
        story = [self.word_to_index[self.story_start]]
        # Predict next word until end token is received
        while not story[-1] == self.word_to_index[self.story_end]:
            num_predictions += 1
            next_word_probs = self.forward_propagate(story)
            samples = np.random.multinomial(1, next_word_probs[-1])
            sampled_word = np.argmax(samples)
            # We don't want the unknown token to appear in the story
            while sampled_word == self.word_to_index[self.unknown_token]:
                num_unknowns += 1
                samples = np.random.multinomial(1, next_word_probs[-1])
                sampled_word = np.argmax(samples)
            # End of while loop
            story.append(sampled_word)
        story_str = [self.index_to_word[word] for word in story[1:-1]]
        story_str = " ".join(story_str)
        return (story_str, (num_unknowns / num_predictions) * 100)
    # End of generate_story()

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
    arg_parse.add_argument("-d", "--dir", default="./grurnn",
                           help="Directory for storing logs.")
    arg_parse.add_argument("-f", "--filename", default="grurnn.log",
                           help="Name of the log file to use.")
    arg_parse.add_argument("-e", "--epochs", default=10, type=int,
                           help="Number of epochs for which to train the RNN.")
    arg_parse.add_argument("-m", "--max", default=None, type=int,
                           help="The maximum number of examples to train on.")
    arg_parse.add_argument("-p", "--patience", default=100000, type=int,
                           help="Number of examples to train before evaluating"
                                + " loss.")
    arg_parse.add_argument("-t", "--test", action="store_true",
                           help="Treat run as test, do not save models")
    arg_parse.add_argument("-l", "--learn_rate", default=0.005, type=float,
                           help="The learning rate to be used in training.")
    arg_parse.add_argument("-o", "--model", default=None,
                           help="Previously trained model to load on init.")
    arg_parse.add_argument("-a", "--anneal", type=float, default=0.00001,
                           help="Sets the minimum possible learning rate.")
    arg_parse.add_argument("-s", "--dataset", default="./datasets/stories.pkl",
                           help="The path to the dataset to be used in "
                                " training.")
    arg_parse.add_argument("-r", "--truncate", type=int, default=100,
                           help="The backpropagate truncate value.")
    arg_parse.add_argument("-v", "--voc_size", type=int, default=8000,
                           help="The vocabulary size for the network.")
    args = arg_parse.parse_args()

    argsdir = args.dir + "/" + time.strftime("%d%m%y%H") + "/";
    sentenceDir = argsdir
    modelDir = argsdir
    logDir = argsdir
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
    testlog.info("Running a new GRU-RNN with logging")

    RNN = GruRNN(model=args.model, trunc=args.truncate, voc_size=args.voc_size)
    RNN.load_data(args.dataset)
    #loss = RNN.calculate_loss(RNN.x_train, RNN.y_train)
    #self.log.info(loss)
    RNN.train_rnn(
        epochs=args.epochs,
        patience=args.patience,
        path=modelDir+"reladred",
        max=args.max,
        testing=args.test,
        learning_rate=args.learn_rate,
        anneal=args.anneal
    )

    if args.test:
        testlog.info("Finished running test.")
        sys.exit(0)

    testlog.info("Generating stories")

    file = open(sentenceDir+"stories.txt", "w")

    attempts = 0
    successes = 0
    percent_unknowns = 0

    while successes < 25:
        story = RNN.generate_story()
        print(story[0])
        percent_unknowns += story[1]
        # if len(story) >= 50:
        file.write(" ".join(story[0]) + "\n")
        successes += 1
        attempts += 1

    file.close()

    testlog.info("Generated %d stories after %d attempts." %
                 (successes, attempts))

    percent_unknowns = percent_unknowns / attempts
    testlog.info("%f percent of the words generated were unknown tokens." %
                 percent_unknowns)
