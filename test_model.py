import grurnn as RNN
import logging
import logging.handlers

testlog = logging.getLogger("TEST")
testlog.setLevel(logging.INFO)

handler = logging.handlers.RotatingFileHandler(
    filename="test.log",
    maxBytes=1024*512,
    backupCount=5
)

formatter = logging.Formatter(
    "%(asctime)s-%(name)s-%(levelname)s-%(message)s"
)

handler.setFormatter(formatter)

testlog.addHandler(handler)
testlog.info("Running a new GRU-RNN with logging")

Network = RNN.GruRNN(model="./grurnn/08031720/models/reladred10.pkl")
testlog.info("Printing self.index_to_word")
testlog.info(Network.index_to_word)

file = open("sentences.txt", "w")

attempts = 0
sents = 0

while sents < 10:
    sentence = Network.generate_sentence()
    file.write(" ".join(sentence) + "\n")
    sents += 1

file.close()

testlog.info("Generated %d sentences after %d attempts." % (sents, attempts))
