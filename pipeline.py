import pickle

from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

from feature_extraction import get_features

annotated_corpus = open('Corpus/annotated_corpus.pickle', 'r')
annotated_corpus = pickle.load(annotated_corpus)

def main():
	gnb = GaussianNB()
	steps = [
		('vectorizer', get_features()),
		('classifier', gnb)
	]
	pipeline = Pipeline(steps)
	y_pred = pipeline.fit([e['question'] for e in annotated_corpus],
						  [e['target'] for e in annotated_corpus])

	while True:
		line = raw_input("Enter a question or STOP\n>>> ")
		if line == 'STOP':
			break
		print y_pred.predict([line])


if __name__ == '__main__':
	main()