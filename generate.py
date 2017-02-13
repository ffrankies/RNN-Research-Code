import vanillarnn as VanillaRnn
import grurnn as GruRNN
import os
import sys
import io

###############################################################################
# Initializing RNN, Loading model
###############################################################################
RNN = GruRNN(model="models/grureladred9.pkl")
print("Finished loading model...")

###############################################################################
# Generating sentences, saving them in file
###############################################################################
with open("grusentencesL.txt", "w") as outFile:

    num_generated = 0
    attempt = 0
    while num_generated < 50:
        sentence = RNN.generate_sentence()
        print("Attempt %d: %s" % (attempt, " ".join(sentence)))
        sys.stdout.flush()
        if len(sentence) > 5:
            # print(" ".join(sentence))
            outFile.write(u' '.join(sentence).encode('utf-8') + "\n")
            sys.stdout.flush()
            num_generated += 1
        attempt += 1

    outFile.close()
