import pickle

f = open("quepy_extraction.dat", 'r')
lines = f.read()
f.close()
lines = lines.split('\n')
result = []
for line in lines:
	if not line:
		continue
	line = line.split('\t')
	try:
		question = {
			'question' : line[0],
			'target' : line[1],
			'recognized' : line[2] == 's'
		}
	except IndexError:
		continue
	result.append(question)

f = open("original_quepy_corpus.pickle", 'w')
pickle.dump(result, f)
f.close()