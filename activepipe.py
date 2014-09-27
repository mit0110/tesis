import argparse
import pickle

from collections import defaultdict
from math import log
from random import choice
from sklearn.metrics import precision_score, classification_report
from sklearn.pipeline import Pipeline
from termcolor import colored

from defaults import default_config


class ActivePipeline(object):
    """Pipeline for a classifier with the following steps:
        1 - Training the classifier from an training corpus
        2 - Getting new evidence from the oracle:
            2.1 - Selecting a question from the unlabeled corpus to pass to
            the oracle.
            2.2 - Re-training with the new information.
    """

    def __init__(self, session_filename='', emulate=False, **kwargs):
        self.filename = session_filename
        self.emulate = emulate
        default_config.update(kwargs)
        self.config = default_config
        self.classifier = self.config['classifier']()
        self.features = self.config['features']
        self.steps = [
            ('features', self.features),
            ('classifier', self.classifier),
        ]
        self.pipeline = Pipeline(self.steps)
        self._get_corpus()
        self.test_corpus = None
        self.user_vectors = []
        self.user_targets = []
        self.user_features = defaultdict(lambda : [])
        self.load_session()
        self.pipeline.partial_fit(self.training_vectors, training_target)
        self.new_instances = 0
        self.classes = self.classifier.classes_.tolist()  # No todos los clasificadores
        # van a tener esto.

    def _train(self):
        self.pipeline.partial_fit(self.user_vectors[-self.new_instances:],
                                  self.user_targets[-self.new_instances:])
        self.new_instances = 0

    def _get_corpus(self):
        training_corpus_f = open(self.config['training_corpus_f'], 'r')
        self.training_corpus = pickle.load(training_corpus_f)
        training_corpus_f.close()
        self.training_vectors = [e['question'] for e in self.training_corpus]
        self.training_target = [e['target'] for e in self.training_corpus]

        unlabeled_corpus_f = open(self.config['u_corpus_f'], 'r')
        # A list of dictionaries
        self.unlabeled_corpus = pickle.load(unlabeled_corpus_f)
        unlabeled_corpus_f.close()

    def _get_test_corpus(self):
        if self.test_corpus:
            return
        test_corpus_f = open(self.config['test_corpus_f'], 'r')
        self.test_corpus = pickle.load(test_corpus_f)
        test_corpus_f.close()
        self.test_vectors = [e['question'] for e in self.test_corpus]
        self.test_targets = [e['target'] for e in self.test_corpus]

    def predict(self, question):
        return self.pipeline.predict(question)

    def _process_prediction(self, instance, prediction):
        """Retrains the classfier with the new instance and prediction."""
        self.new_instances += 1
        self.user_vectors.append(instance)
        self.user_targets.append(prediction)
        self._train()
        return True

    def _most_probable_classes(self, predicted_classes):
        indexes = predicted_classes.argsort()
        result = []
        indexes = indexes[0].tolist()
        indexes.reverse()
        for index in indexes[:self.config['number_of_options']]:
            result.append(self.classes[index])  # Class name
        result.append(self.classes[-1])
        return result

    def instance_bootstrap(self, get_class):
        """Presents a new question to the user until the answer is 'stop'.

        Args:
            get_class: A function that takes an instance and a list of possible
            classes. Returns the correct class for the instance.
        """
        while True:
            new_instance = self.get_next_instance()
            if self.emulate and 'target' in new_instance:
                new_instance = new_instance['question']
                prediction = new_instance['target']
                message = "Adding instance {}, {}".format(new_instance, prediction)
                print colored(message, "red")
            if not self.emulate or 'target' not in new_instance:
                new_instance = new_instance['question']
                classes = self.pipeline.predict_log_proba([new_instance])
                classes = self._most_probable_classes(classes)
                prediction = get_class(new_instance, classes)
            if prediction == 'stop':
                break
            self._process_prediction(new_instance, prediction)

    def feature_boostrap(self, get_features_4_class):
        """Presents a class and possible features until the predictio is stop.

        Args:
            get_features_4_class: A function that receives a class and a list
            of features. It must return a list of features associated with the
            class.
        """
        # Re train to get different classes.
        # Don't ask for the same feature twice. Or not?
        stop = False
        while not stop:
            class_number, feature_numbers = self.get_class_and_features()
            class_name = self.classes[class_number]
            feature_names = [self.features.column_to_feature(pos)[1]
                             for pos in feature_numbers]
            prediction = get_features_4_class(class_name, feature_names)
            if prediction == 'stop':
                break
            prediction = [feature_names.index(f) for f in prediction]
            self.user_features[class_name] += prediction

            # Modify the classifier
            for feature in prediction:
                self.classifier.feature_log_prob_[class_number][feature] += \
                    log(self.config['alpha'])

            self._train()

    def get_next_instance(self):
        """Selects an instance to be sent to the user.

        This is the core of the active learning algorithm.

        Returns:
            An instance selected from the unlabeled_corpus.
        """
        try:
            question = choice(self.unlabeled_corpus)
            return question
        except IndexError:
            return None

    def get_class_and_features(self):
        """Selects a class and a list of features to be sent to the oracle.

        Returns:
            A tuple where the first element is a class number and the second
            one a list of features numbers.
        """
        feature_prob = self.classifier.feature_log_prob_
        selected_class = feature_prob.sum(axis=1).argmax()
        selected_f_pos = feature_prob[selected_class].argsort()[-10:]
        return selected_class, selected_f_pos

    def evaluate_test(self):
        """Evaluates the classifier with the testing set.
        """
        self._get_test_corpus()
        return self.pipeline.score(self.test_vectors, self.test_targets)

    def evaluate_training(self):
        """Evaluate the accuracy of the classifier with the labeled data.
        """
        # Agregamos la evidencia del usuario para evaluacion?
        return self.pipeline.score(self.training_vectors, self.training_target)

    def get_report(self):
        """
        Returns:
            A sklearn.metrics.classification_report on the performance
            of the cassifier over the test corpus.
        """
        self._get_test_corpus()
        predicted_targets = self.predict(self.test_vectors)
        return classification_report(self.test_targets, predicted_targets)

    def label_corpus(self):
        """Saves the classes of unlabeled vectors in the unlabeled_corpus file.

        Returns:
            The number of new classes added to the corpus.
        """
        for index, instances in enumerate(self.user_vectors):
            u_question = next((q for q in self.unlabeled_corpus
                              if q['question'] == instances), None)
            if u_question and u_question['question'] == instances:
                u_question['target'] = self.user_targets[index]
        # Save file
        f = open(self.config['u_corpus_f'], 'w')
        pickle.dump(self.unlabeled_corpus, f)
        f.close()
        return len(self.user_vectors)

    def save_session(self, filename):
        """Saves the instances and targets introduced by the user in filename.

        Writes a pickle tuple in the file that can be recovered using the
        method load_session.

        Returns:
            False in case of error, True in case of success.
        """
        if not self.user_vectors or not filename:
            return False

        f = open(filename, 'w')
        to_save = (self.user_vectors, self.user_targets)
        pickle.dump(to_save, f)
        f.close()
        return True

    def load_session(self):
        """Loads the instances and targets stored on filename.

        Overwrites the previous answers of the users.

        Args:
            filename: a string. The name of a file that has a  pickle tuple.
            The first element of the tuple is a list of vectors, the second is
            a list of targets.

        Returns:
            False in case of error, True in case of success.
        """
        if not self.filename:
            return False
        f = open(self.filename, 'r')
        self.user_vectors, self.user_targets = pickle.load(f)
        f.close()
        return True

