import vanillarnn as RNN

Network = RNN.VanillaRNN(model="./Vanilla/models/17021713/reladred19.pkl")

file = open("sentences.txt", "w")

attempts = 0
sents = 0

while sents < 10:
    sentence = Network.generate_sentence()
    if len(sentence) >= 5:
        file.write(" ".join(sentence) + "\n")
        sents += 1
    attempts += 1

file.close()

print("Generated %d sentences after %d attempts." % (sents, attempts))
