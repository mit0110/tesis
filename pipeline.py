import pickle

from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

from feature_extraction import get_features



class ClassifierPipeline(object):
	"""Pipeline for a classifier with the following steps:
		1 - Training the classifier from an annotated corpus
		2 - Getting new evidence from the oracle:
			2.1 - Selecting a question from the unlabeled corpus to pass to
			the oracle.
			2.2 - Re-training with the new information.
	"""

	def __init__(self):
		gnb = GaussianNB()
		self.steps = [
			('vectorizer', get_features()),
			('classifier', gnb),
		]
		self._get_corpus()
		self.pipeline = Pipeline(self.steps)
		self._train()
		self.test_corpus = None
		
	def _train(self):
		self.predictor = self.pipeline.fit(
			self.annotated_question,
			self.annotated_target
		)

	def _get_corpus(self):
		annotated_corpus_f = open('corpus/annotated_corpus.pickle', 'r')
		self.annotated_corpus = pickle.load(annotated_corpus_f)
		annotated_corpus_f.close()
		self.annotated_question = [e['question'] for e in self.annotated_corpus]
		self.annotated_target = [e['target'] for e in self.annotated_corpus]

		unlabeled_corpus_f = open('corpus/unlabeled_corpus.pickle', 'r')
		self.unlabeled_corpus = pickle.load(unlabeled_corpus_f)
		unlabeled_corpus_f.close()

	def _get_test_corpus(self):
		if self.test_corpus:
			return
		test_corpus_f = open('corpus/test_corpus.pickle', 'r')
		self.test_corpus = pickle.load(test_corpus_f)
		test_corpus_f.close()
		self.test_questions = [e['question'] for e in self.test_corpus]
		self.test_targets = [e['target'] for e in self.test_corpus]
		
	def predict(self, question):
		return self.predictor.predict(question)

	def _process_answer(self, answer, question, prediction):
		"""
		"""
		if answer is 'y':
			self.annotated_question.append(question)
			self.annotated_target.append(prediction)

	def bootstrap(self):
		"""
		"""
		message = "Is the same as {}? y/n/d/STOP\n>>> "
		while True:
			new_question = self.get_next_question()
			print new_question
			prediction = str(self.predict([new_question]))
			line = raw_input(message.format(prediction))
			if line == 'STOP':
				print "Thank you for your time!"
				self._train()
				print "New accuracy: TRAINING {}% - TEST {}%\n".format(
					self.evaluate_training(), self.evaluate_test()
				)
				break
			self._process_answer(line, new_question, prediction)

	def get_next_question(self):
		return self.unlabeled_corpus.pop()

	def _get_accuracy(self, predicted_targets, real_targets):
		ok = 0
		index = 0
		for predicted_target in predicted_targets:
			if predicted_target == real_targets[index]:
				ok += 1
			index += 1
		return ok*100/len(predicted_targets)

	def evaluate_test(self):
		"""Evaluates the classifier with the testing set.
		"""
		self._get_test_corpus()
		predicted_targets = self.predict(self.test_questions)
		return self._get_accuracy(predicted_targets, self.test_targets)

	def evaluate_training(self):
		"""Evaluate the accuracy of the classifier with the labeled data.
		"""
		# Agregamos la evidencia del usuario para evaluacion?
		predicted_targets = self.predict(self.annotated_question)
		return self._get_accuracy(predicted_targets, self.annotated_target)


def main():
	pipe = ClassifierPipeline()
	pipe.bootstrap()


if __name__ == '__main__':
	main()