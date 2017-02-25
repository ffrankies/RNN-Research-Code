###############################################################################
# Source: RNN tutorial from www.wildml.com
#
# The following is a python3 implementation of a  Recurrent Neural Network with
# GRU units to predict the next word in a sentence. It uses the
# RMSprop algorithm to speed up converge with gradient descent. The nature of
# this network is that it can be used to generate sentences that will resemble
# the format of the training set. This particular version was meant to be used
# with reddit comment datasets, although it should in theory work fine with
# other sentence-based datasets, as well.
# The various datasets that this particular RNN will be used for will be
# kernels of the main dataset of reddit comments from May 2015 fourd on kaggle
# Dataset source: https://www.kaggle.com/reddit/reddit-comments-may-2015
#
# Author: Frank Derry Wanye
# Date: 10 Feb 2017
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
import operator
import logging
import logging.handlers
import argparse

class RMSpropRNN(object):
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
    :date: 10 Feb 2017
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

        self.log = logging.getLogger("TEST.RMSprop")
        self.log.setLevel(logging.INFO)

        self.unknown_token = "UNKNOWN_TOKEN"
        self.sentence_start_token = "SENTENCE_START"
        self.sentence_end_token = "SENTENCE_END"

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

            self.cache_ih = theano.shared(
                name='cache_ih',
                value=np.zeros(weights_ih.shape).astype(theano.config.floatX))

            self.cache_hh = theano.shared(
                name='cache_hh',
                value=np.zeros(weights_hh.shape).astype(theano.config.floatX))

            self.cache_ho = theano.shared(
                name='cache_ho',
                value=np.zeros(weights_ho.shape).astype(theano.config.floatX))

            self.cache_bias = theano.shared(
                name='cache_bias',
                value=np.zeros(bias.shape).astype(theano.config.floatX))

            self.cache_out_bias = theano.shared(
                name='cache_out_bias',
                value=np.zeros(out_bias.shape).astype(theano.config.floatX))

            self.d_weights_ih = theano.shared(
                name='d_weights_ih',
                value=np.zeros(weights_ih.shape).astype(theano.config.floatX))

            self.d_weights_hh = theano.shared(
                name='d_weights_hh',
                value=np.zeros(weights_hh.shape).astype(theano.config.floatX))

            self.d_weights_ho = theano.shared(
                name='d_weights_ho',
                value=np.zeros(weights_ho.shape).astype(theano.config.floatX))

            self.d_bias = theano.shared(
                name='d_bias',
                value=np.zeros(bias.shape).astype(theano.config.floatX))

            self.d_out_bias = theano.shared(
                name='d_out_bias',
                value=np.zeros(out_bias.shape).astype(theano.config.floatX))

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

                self.weights_ih = params[3]
                self.weights_hh = params[4]
                self.weights_ho = params[5]

                self.vocabulary = params[6]
                if not self.vocabulary[-1] == self.unknown_token:
                    self.log.info("Appending unknown token")
                    self.vocabulary[-1] = self.unknown_token
                self.index_to_word = params[7]
                self.word_to_index = params[8]

                self.bias = params[9]
                self.out_bias = params[10]

                self.cache_ih = params[11]
                self.cache_hh = params[12]
                self.cache_ho = params[13]
                self.cache_bias = params[14]
                self.cache_out_bias = params[15]
        # End of if statement

        # Symbolic representation of one input sentence
        input = T.ivector('isentence')

        # Symbolic representation of the one output sentence
        output = T.ivector('osentence')

        # Symbolic representation of the cache decay for RMSprop
        decay = T.scalar('decay')

        # Stochastic Gradient Descent step
        learning_rate = T.scalar('learning_rate')

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

            current_state = (T.ones_like(update_gate) - update_gate) * hypothesis + update_gate * previous_state

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
            outputs_info=[None, dict(initial=T.zeros(self.hidden_size))],
            name="forward_propagate"
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
        d_bias = T.grad(out_error, self.bias)
        d_out_bias = T.grad(out_error, self.out_bias)

        # Symbolic theano functions
        self.forward_propagate = theano.function([input], out,
            name="forward_propagate")
        self.predict = theano.function([input], prediction, name="predict")
        self.calculate_error = theano.function([input, output], out_error,
            name="calculate_error")
        self.bptt = theano.function([input, output],
            [d_weights_ih, d_weights_hh, d_weights_ho, d_bias, d_out_bias],
            name="bptt")

        # RMSprop parameters
        cache_ih = (decay * self.cache_ih) + ((1 - decay) * T.sqr(d_weights_ih))
        cache_hh = (decay * self.cache_hh) + ((1 - decay) * T.sqr(d_weights_hh))
        cache_ho = (decay * self.cache_ho) + ((1 - decay) * T.sqr(d_weights_ho))
        cache_bias = (decay * self.cache_bias) + ((1 - decay) * T.sqr(d_bias))
        cache_out_bias = (decay * self.cache_out_bias) + ((1 - decay) * T.sqr(d_out_bias))
        eps = 1e-6 # Prevents division by 0

        self.sgd_step = theano.function(
            [input, output, learning_rate, theano.In(decay, value=0.9)],
            [],
            updates=[
                (self.weights_ih, self.weights_ih - learning_rate *
                 d_weights_ih / (T.sqrt(self.cache_ih + eps))),
                (self.weights_hh, self.weights_hh - learning_rate *
                 d_weights_hh / (T.sqrt(self.cache_hh + eps))),
                (self.weights_ho, self.weights_ho - learning_rate *
                 d_weights_ho / (T.sqrt(self.cache_ho + eps))),
                (self.bias, self.bias - learning_rate * d_bias /
                 (T.sqrt(self.cache_bias + eps))),
                (self.out_bias, self.out_bias - learning_rate *
                 d_out_bias / (T.sqrt(self.cache_out_bias + eps))),
                (self.cache_ih, cache_ih),
                (self.cache_hh, cache_hh),
                (self.cache_ho, cache_ho),
                (self.cache_bias, cache_bias),
                (self.cache_out_bias, cache_out_bias),
                (self.d_weights_ih, d_weights_ih),
                (self.d_weights_hh, d_weights_hh),
                (self.d_weights_ho, d_weights_ho),
                (self.d_bias, d_bias),
                (self.d_out_bias, d_out_bias)],
            mode=theano.compile.MonitorMode(
                post_func=theano.compile.monitormode.detect_nan
            )
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
                  path=None, max=None, testing=False, anneal=True, decay=0.9):
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
                    used in the training
        """
        if self.x_train is None or self.y_train is None:
            self.log.error("Need to load data before training the rnn")
            return False

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
                                  learning_rate, decay)
                    examples_seen += 1
                    self.log.debug("Examples seen: %d" % examples_seen)
                    if examples_seen % patience == 0:
                        self.log.info("Evaluated %d examples" % examples_seen)

            else:
                for example in range(len(self.y_train[:max])):
                    self.sgd_step(self.x_train[example], self.y_train[example],
                                  learning_rate, decay)
                    examples_seen += 1
                    self.log.debug("Examples seen: %d" % examples_seen)
                    if examples_seen % patience == 0:
                        self.log.info("Evaluated %d examples" % examples_seen)
            # End of training for epoch

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
            if (anneal and len(losses) > 1 and losses[-1][1] > losses[-2][1]):
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
                params = (
                    self.vocabulary_size, self.hidden_size, self.bptt_truncate,
                    self.weights_ih, self.weights_hh, self.weights_ho,
                    self.vocabulary, self.index_to_word, self.word_to_index,
                    self.bias, self.out_bias, self.cache_ih, self.cache_hh,
                    self.cache_ho, self.cache_bias, self.cache_out_bias
                )

                if path == None:
                    modelPath = "models/reladred" + str(epoch) + ".pkl"
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
        if self.word_to_index is None:
            self.log.error("Need to load a model or data before this step.")
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

    def generate_sentence_seed(self, seed):
        """
        Generates one sentence based on current model parameters. Model needs
        to be loaded or trained before this step in order to produce any
        results.

        :return type: list of strings
        :return param: a generated sentence, with each word being an item in
                       the array.
        """
        if self.word_to_index is None:
            self.log.error("Need to load a model or data before this step.")
            return []

        if seed is None:
            self.log.error("There is not sentence seed.")
            return []

        # Start sentence with the start token
        sentence = [self.word_to_index[self.sentence_start_token]]
        sentence.extend(seed.lower().split(' '))

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

    def grad_check(self):
        """
        NOTE: CudeNDArrays returned instead of actual arrays in some cases.
        Performs a gradient descent check on this network.
        """
        if self.x_train is None or self.y_train is None:
            self.log.error("No training data provided.")
            return False

        h=0.001
        error_threshold=0.01

        # Overwrite the bptt attribute.
        self.bptt_truncate = 1000
        # Calculate the gradients using backprop
        bptt_gradients = self.bptt(self.x_train[0], self.y_train[0])
        # List of all parameters we want to chec.
        model_parameters = ['weights_ih', 'weights_hh', 'weights_ho',
                            'bias', 'out_bias']
        # Gradient check for each parameter
        for pidx, pname in enumerate(model_parameters):
            # Get the actual parameter value from the mode, e.g. model.W
            parameter_T = operator.attrgetter(pname)(self)
            parameter = parameter_T.get_value()
            self.log.info("Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape)))
            # Iterate over each element of the parameter matrix
            it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                ix = it.multi_index
                # Save the original value so we can reset it later
                original_value = parameter[ix]
                # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
                parameter[ix] = original_value + h
                parameter_T.set_value(parameter)
                gradplus = self.calculate_total_loss(
                    [self.x_train[0]],[self.y_train[0]])
                parameter[ix] = original_value - h
                parameter_T.set_value(parameter)
                gradminus = self.calculate_total_loss(
                    [self.x_train[0]],[self.y_train[0]])
                estimated_gradient = (gradplus - gradminus)/(2*h)
                parameter[ix] = original_value
                parameter_T.set_value(parameter)
                # The gradient for this parameter calculated using backpropagation
                backprop_gradient = bptt_gradients[pidx][ix]
                # calculate The relative error: (|x - y|/(|x| + |y|))
                relative_error = np.abs(backprop_gradient - estimated_gradient)/(np.abs(backprop_gradient) + np.abs(estimated_gradient))
                # If the error is to large fail the gradient check
                if relative_error > error_threshold:
                    self.log.error("Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix))
                    self.log.error("+h Loss: %f" % gradplus)
                    self.log.error("-h Loss: %f" % gradminus)
                    self.log.error("Estimated_gradient: %f" % estimated_gradient)
                    self.log.error("Backpropagation gradient: %f" % backprop_gradient)
                    self.log.error("Relative Error: %f" % relative_error)
                    return
                it.iternext()
            self.log.info("Gradient check for parameter %s passed." % (pname))
    # End of gradient_check()

    def update_weight_ratio(self):
            """
            Finds the ratio of updates to weights being done by the network.
            According to Karpathy, this ratio should be ~1e-3 (0.001). If it is
            lower than that, learning rate might be too low, and vice versa.

            NOTE: Once the RMSprop momentum algorithm is set up, that should be
            used in here instead of vanilla sgd. Also, this can be built into the
            forward_propagate or the back_propagate algorithm, maybe.
            """
            # Ratio for weights from input to hidden layer
            param_scale_ih = np.linalg.norm(self.weights_ih.ravel())
            ih_update = -learning_rate * self.d_wght_ih
            update_scale_ih = np.linalg.norm(ih_update.ravel())
            self.log.info("updates:weights for weights_ih: %f" %
                          (update_scale_ih / param_scale_ih))
            # Ratio for weights from hidden layer to hidden layer
            param_scale_hh = np.linalg.norm(self.weights_hh.ravel())
            hh_update = -learning_rate * self.d_wght_hh
            update_scale_hh = np.linalg.norm(hh_update.ravel())
            self.log.info("updates:weights for weights_hh: %f" %
                          (update_scale_hh / param_scale_hh))
            # Ratio for weights from hidden layer to output layer
            param_scale_ho = np.linalg.norm(self.weights_ho.ravel())
            ho_update = -learning_rate * self.d_wght_ho
            update_scale_ho = np.linalg.norm(ho_update.ravel())
            self.log.info("updates:weights for weights_hh: %f" %
                          (update_scale_ho / param_scale_ho))
    # End of weight_update_ratio()

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
    arg_parse.add_argument("-d", "--dir", default="./rmsrnn",
                           help="Directory for storing logs.")
    arg_parse.add_argument("-f", "--filename", default="rmsrnn.log",
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
    arg_parse.add_argument("-a", "--anneal", action="store_false",
                           help="Set this option to not anneal learning rate.")
    arg_parse.add_argument("-c", "--decay", default=0.9, type=float,
                           help="The decay to be applied to RMSprop.")
    arg_parse.add_argument("-g", "--grad_check", action="store_true",
                           help="Indicates a gradient check.")
    arg_parse.add_argument("-s", "--seed",
                           help="The seed phrase for generating sentences.")
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
    testlog.info("Initializing a new RMSprop-RNN with logging")

    RNN = RMSpropRNN(model=args.model)
    RNN.load_data()

    if args.grad_check:
        testlog.info("Running a gradient check.")
        RNN.grad_check()
        sys.exit()

    if args.seed is not None:
        testlog.info("Generating a sentence with given seed.")
        RNN.generate_sentence_seed(args.seed);
        sys.exit()

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
