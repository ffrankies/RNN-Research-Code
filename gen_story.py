import grurnn

RNN = grurnn.GruRNN(model='./grurnn/08031721/models/reladred20.pkl')

with open("stories.txt", "w", encoding="utf-8") as file:
	num = 0
	while num < 25:
		print("----NUM%d----" % num)	
		story = RNN.generate_story()
		print("Length of story: %d" % len(story))
		file.write(" ".join(story))
		num += 1


