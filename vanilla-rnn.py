###############################################################################
# Source: RNN tutorial from www.wildml.com
#
# The following is a python3 implementation of a vanilla Recurrent Neural
# Network used to predict the next word in a sentence. The nature of this
# network is that it can be used to generate sentences that will resemble
# the format of the training set. This particular version was meant to be used
# with reddit comment datasets, although it should in theory work fine with
# other sentence-based datasets, as well.
# The various datasets that this particular RNN will be used for will be
# kernels of the main dataset of reddit comments from May 2015 fourd on kaggle
# Dataset source: https://www.kaggle.com/reddit/reddit-comments-may-2015
#
# Author: Frank Derry Wanye
# Date: 25 Nov 2016
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
import timeit

class VanillaRNN(object):
    """
    A Recurrent Neural Network that does not implement some of the more
    complex techniques for combating vanishing and exploding gradients, like
    the LSTM. The network essentially looks like this:
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
    :date: 26 Nov 2016
    """

    def __init__(self, vocabulary_size=8000, hidden_size=100, bptt_truncate=4):
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
        """
        print("Initializing RNN parameters and functions...")

        self.vocabulary_size = vocabulary_size
        self.hidden_size = hidden_size
        self.bptt_truncate = bptt_truncate

        # Instantiate the network weights
        # I feel like the first and third are switched for some reason...
        # but it's pretty consistent in the example code. Perhaps it's
        # backwards for a purpose
        # The weights going from input layer to hidden layer (U, in tutorial)
        weights_ih = np.random.uniform(-np.sqrt(1./vocabulary_size),
                                        np.sqrt(1./vocabulary_size),
                                        (hidden_size, vocabulary_size))
        # The weights going from hidden layer to hidden layer (W, in tutorial)
        weights_hh = np.random.uniform(-np.sqrt(1./vocabulary_size),
                                        np.sqrt(1./vocabulary_size),
                                        (hidden_size, hidden_size))
        # The weights going from hidden layer to output layer (V, in tutorial)
        weights_ho = np.random.uniform(-np.sqrt(1./vocabulary_size),
                                        np.sqrt(1./vocabulary_size),
                                        (vocabulary_size, hidden_size))

        self.weights_ih = theano.shared(
            name='weights_ih', value=weights_ih.astype(theano.config.floatX))

        self.weights_hh = theano.shared(
            name='weights_hh', value=weights_hh.astype(theano.config.floatX))

        self.weights_ho = theano.shared(
            name='weights_ho', value=weights_ho.astype(theano.config.floatX))

        # Symbolic representation of one input sentence
        input = T.ivector('sentence')

        # Symbolic representation of the one output sentence
        output = T.ivector('sentence')

        def forward_propagate(word, previous_state, ih=None, hh=None, ho=None):
            """
            Vertically propagates one of the words.

            :type word: int
            :param word: the index of the current input word

            :type previous_state: T.dvector()
            :param word: the output of the hidden layer from the previous
                         horizontal layer
            """
            if ih == None:
                ih = self.weights_ih
            if hh == None:
                hh = self.weights_hh
            if ho == None:
                ho = self.weights_ho
            current_state = T.tanh(ih[:, word] + hh.dot(previous_state))
            current_output = T.nnet.softmax(ho.dot(current_state))
            # Not sure why current_output[0] and not just current_output...
            return [current_output[0], current_state]

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
            outputs_info=[None, dict(initial=T.zeros(self.hidden_size))],
            non_sequences=[self.weights_ih, self.weights_hh, self.weights_ho],
            truncate_gradient=self.bptt_truncate,
            strict=True
        )

        # Predicts the output words for each word in the sentence
        prediction = T.argmax(out, axis=1)

        # Calculates the output error between the predicted output and the
        # actual output
        out_error = T.sum(T.nnet.categorical_crossentropy(out, output))

        # Symbolically represents gradient calculations for gradient descent
        d_weights_ih = T.grad(out_error, self.weights_ih)
        d_weights_hh = T.grad(out_error, self.weights_hh)
        d_weights_ho = T.grad(out_error, self.weights_ho)

        # Symbolic theano functions
        self.forward_propagate = theano.function([input], out)
        self.predict = theano.function([input], prediction)
        self.calculate_error = theano.function([input, output], out_error)
        self.bptt = theano.function([input, output],
            [d_weights_ih, d_weights_hh, d_weights_ho])

        # Stochastic Gradient Descent step
        learning_rate = T.scalar('learning_rate')
        self.sgd_step = theano.function(
            [input, output, learning_rate], [],
            updates=[(self.weights_ih, self.weights_ih - learning_rate *
                      d_weights_ih),
                     (self.weights_hh, self.weights_hh - learning_rate *
                      d_weights_hh),
                     (self.weights_ho, self.weights_ho - learning_rate *
                      d_weights_ho)]
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
        print("Calculating average categorical crossentropy loss...")
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
        print("Loading the dataset from %s" % filePath)
        file = open(filePath, "rb")
        vocabulary, index_to_word, word_to_index, x_train, y_train = cPickle.load(file)

        print("Dataset contains %d words" % len(vocabulary))

        self.vocabulary = vocabulary
        self.index_to_word = index_to_word
        self.word_to_index = word_to_index
        self.x_train = x_train
        self.y_train = y_train
    # End of calculate_loss()

    def train_rnn(self, learning_rate=0.005, epochs=1, patience=10000, path=None):
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
        """
        if self.x_train == None or self.y_train == None:
            print("Need to load data before training the rnn")
            return

        # Keep track of losses so that they can be plotted
        start_time = timeit.default_timer()

        losses = []
        examples_seen = 0
        for epoch in range(epochs):
            print("Training the model: epoch %d" % epoch)

            # Train separately for each training example (no need for
            # minibatches)
            for example in range(len(self.y_train)):
                self.sgd_step(self.x_train[example], self.y_train[example],
                              learning_rate)
                examples_seen += 1
                # Evaluate loss after every 'patience' examples
                if examples_seen % patience == 0:
                    print("Evaluating loss: example %d" % examples_seen)
                    loss = self.calculate_loss(self.x_train, self.y_train)
                    losses.append((examples_seen, loss))
                    print("RNN incurred a loss of %f after %d examples" %
                          (loss, examples_seen))
                    # Adjust learning rate if loss increases
                    if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                        learning_rate = learning_rate * 0.5
                        print("Setting learning rate to %f" % learning_rate)
                # End of loss evaluation
            # End of training for epoch

            # Saving model parameters
            params = (
                self.vocabulary_size, self.hidden_size, self.bptt_truncate,
                self.weights_ih, self.weights_hh, self.weights_ho,
                self.vocabulary, self.index_to_word, self.word_to_index
            )
            if path == None:
                with open("models/reladred" + epoch + ".pkl", "wb") as file:
                    cPickle.dump(params, file, protocol=2)
            else:
                with open(path + epoch + ".pkl", "wb") as file:
                    cPickle.dump(params, file, protocol=2)
        # End of training

        end_time = timeit.default_timer()

        print(
            ("Finished training the rnn for %d epochs, with a final loss of %f."
             + " Training took %.2f m") %
            (epochs, losses[-1][1], (end_time - start_time) / 60)
        )
    # End of train_rnn()

if __name__ == "__main__":
    RNN = VanillaRNN()
    RNN.load_data()
    #loss = RNN.calculate_loss(RNN.x_train, RNN.y_train)
    #print(loss)
    RNN.train_rnn(epochs=10)
