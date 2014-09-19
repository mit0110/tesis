import argparse
import pickle
import sys

from feature_extraction import get_features
from random import choice
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion
from termcolor import colored


NUMBER_OF_OPTIONS = 30
U_CORPUS_F = 'corpus/unlabeled_corpus.pickle'
TEST_CORPUS_F = 'corpus/test_corpus.pickle'
TRAINING_CORPUS_F = 'corpus/training_corpus.pickle'
MAX_NGRAMS = 3


class ClassifierPipeline(object):
    """Pipeline for a classifier with the following steps:
        1 - Training the classifier from an training corpus
        2 - Getting new evidence from the oracle:
            2.1 - Selecting a question from the unlabeled corpus to pass to
            the oracle.
            2.2 - Re-training with the new information.
    """

    def __init__(self, session_filename='', emulate=False, label_corpus=False):
        self.filename = session_filename
        self.emulate = emulate
        self.label_corpus = label_corpus
        mnb = MultinomialNB()
        countv = CountVectorizer(ngram_range=(1, MAX_NGRAMS))
        self.steps = [
            ('vect', FeatureUnion([('custom', get_features()),
                                   ('n_grams', countv)])),
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
        training_corpus_f = open(TRAINING_CORPUS_F, 'r')
        self.training_corpus = pickle.load(training_corpus_f)
        training_corpus_f.close()
        self.training_question = [e['question'] for e in self.training_corpus]
        self.training_target = [e['target'] for e in self.training_corpus]

        unlabeled_corpus_f = open(U_CORPUS_F, 'r')
        # A list of dictionaries
        self.unlabeled_corpus = pickle.load(unlabeled_corpus_f)
        unlabeled_corpus_f.close()

    def _get_test_corpus(self):
        if self.test_corpus:
            return
        test_corpus_f = open(TEST_CORPUS_F, 'r')
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
            self.exit()
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
                question = new_question['question']
                target = new_question['target']
                self.user_questions.append(question)
                self.user_answers.append(target)
                self._train()
                message = "Adding question {}, {}".format(question, target)
                print colored(message, "red")
            if not self.emulate or 'target' not in new_question:
                new_question = new_question['question']
                print "*******************************************************"
                print "\nWhat is the correct template? Write the number or STOP\n"
                print colored(new_question, "red", "on_white", attrs=["bold"])
                pred_classes = self.pipeline.predict_log_proba([new_question])
                pred_classes = self._most_probable_classes(pred_classes)
                self._print_classes(pred_classes)
                line = raw_input(">>> ")
                stop = self._process_answer(line, new_question, pred_classes)

    def get_next_question(self):
        try:
            question = choice(self.unlabeled_corpus)
            return question
        except IndexError:
            return None

    def evaluate_test(self):
        """Evaluates the classifier with the testing set.
        """
        self._get_test_corpus()
        return self.pipeline.score(self.test_questions, self.test_targets)

    def evaluate_training(self):
        """Evaluate the accuracy of the classifier with the labeled data.
        """
        # Agregamos la evidencia del usuario para evaluacion?
        return self.pipeline.score(self.training_question, self.training_target)

    def exit(self):
        self._get_test_corpus()
        predicted_targets = self.predict(self.test_questions)
        print "Final Estimation on test corpus"
        print classification_report(self.test_targets, predicted_targets)
        if self.user_questions:
            self.save_session()
        if self.label_corpus:
            for index, question in enumerate(self.user_questions):
                u_question = next((q for q in self.unlabeled_corpus
                                  if q['question'] == question), None)
                if u_question and u_question['question'] == question:
                    u_question['target'] = self.user_answers[index]
            # Save file
            print "Adding {} new questions".format(len(self.user_questions))
            f = open(U_CORPUS_F, 'w')
            pickle.dump(self.unlabeled_corpus, f)
            f.close()

    def save_session(self):
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
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--output_file', required=False,
                        type=str)
    parser.add_argument('-e', '--emulate', action='store_true')
    parser.add_argument('-l', '--label_corpus', action='store_true')
    args = parser.parse_args()
    pipe = ClassifierPipeline(session_filename=args.output_file,
                              emulate=args.emulate,
                              label_corpus=args.label_corpus)
    pipe.bootstrap()


if __name__ == '__main__':
    main()
