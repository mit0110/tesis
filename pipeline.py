import pickle
import sys

from feature_extraction import get_features
from random import choice
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from termcolor import colored


NUMBER_OF_OPTIONS = 30


class ClassifierPipeline(object):
    """Pipeline for a classifier with the following steps:
        1 - Training the classifier from an training corpus
        2 - Getting new evidence from the oracle:
            2.1 - Selecting a question from the unlabeled corpus to pass to
            the oracle.
            2.2 - Re-training with the new information.
    """

    def __init__(self):
        if len(sys.argv) >= 2:
            self.filename = sys.argv[1]
        else:
            self.filename = None
        self.emulate = (len(sys.argv) == 3 and sys.argv[2] == 'emulate')
        mnb = MultinomialNB()
        self.steps = [
            ('vectorizer', get_features()),
            ('classifier', mnb),
        ]
        self._get_corpus()
        self.pipeline = Pipeline(self.steps)
        self.test_corpus = None
        self.user_questions = []
        self.user_answers = []
        self.load_session()
        self._train()
        self.classes = mnb.classes_.tolist()
        
    def _train(self):
        self.pipeline.fit(
            self.training_question + self.user_questions,
            self.training_target + self.user_answers
        )

    def _get_corpus(self):
        training_corpus_f = open('corpus/training_corpus.pickle', 'r')
        self.training_corpus = pickle.load(training_corpus_f)
        training_corpus_f.close()
        self.training_question = [e['question'] for e in self.training_corpus]
        self.training_target = [e['target'] for e in self.training_corpus]

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
        return self.pipeline.predict(question)

    def _process_answer(self, answer, question, predicted_classes):
        """
        """
        if answer.lower() == 'stop':
            print "Thank you for your time!"
            self.save_session()
            return True

        try:
            answer = int(answer)
            prediction = predicted_classes[answer]
        except (ValueError, IndexError):
            print colored("Please insert one of the listed numbers", "red")
            return False

        print colored("Adding result", "green")
        self.user_questions.append(question)
        self.user_answers.append(prediction)
        self._train()
        print "New accuracy: TRAINING {}% - TEST {}%\n".format(
            self.evaluate_training(), self.evaluate_test()
        )
        return False

    def _print_classes(self, predicted_classes):
        message = "{} - {}"
        for (counter, class_name) in enumerate(predicted_classes):
            print message.format(counter, class_name)

    def _most_probable_classes(self, predicted_classes):
        indexes = predicted_classes.argsort()
        result = []
        indexes = indexes[0].tolist()
        indexes.reverse()
        for index in indexes[:NUMBER_OF_OPTIONS]:
            result.append(self.classes[index])  # Class name
        result.append(self.classes[-1])
        return result

    def bootstrap(self):
        """
        """
        stop = False
        while not stop:
            new_question = self.get_next_question()
            if self.emulate and 'target' in new_question:
                self.user_questions.append(new_question['question'])
                self.user_answers.append(new_question['target'])
                self._train()
            else:
                new_question = new_question['question']
                print "*******************************************************"
                print colored(new_question, "red", "on_white", attrs=["bold"])
                pred_classes = self.pipeline.predict_log_proba([new_question])
                pred_classes = self._most_probable_classes(pred_classes)
                print "\nWhat is the correct template? Write the number or STOP\n"
                self._print_classes(pred_classes)
                line = raw_input(">>> ")
                stop = self._process_answer(line, new_question, pred_classes)

    def get_next_question(self):
        try:
            question = choice(self.unlabeled_corpus)
            return question
        except IndexError:
            return None

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
        predicted_targets = self.predict(self.training_question)
        return self._get_accuracy(predicted_targets, self.training_target)

    def save_session(self):
        if self.user_questions:
            filename = raw_input("Insert the filename to save or press enter\n")
            if not filename:
                return
            f = open("sessions/{}".format(filename), 'w')
            to_save = (self.user_questions, self.user_answers)
            pickle.dump(to_save, f)
            f.close()

    def load_session(self):
        if self.filename:
            f = open(self.filename, 'r')
            self.user_questions, self.user_answers = pickle.load(f)
            f.close()
            print "Session {} loaded\n".format(self.filename)


def main():
    pipe = ClassifierPipeline()
    pipe.bootstrap()


if __name__ == '__main__':
    main()
