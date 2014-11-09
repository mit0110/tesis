import argparse
import pickle

from collections import defaultdict
from corpus import Corpus
from math import log
from numpy import array
from random import randint
from scipy.sparse import vstack
from sklearn.metrics import precision_score, classification_report
from termcolor import colored

from defaults import default_config


class ActivePipeline(object):
    """
    """

    def __init__(self, session_filename='', emulate=False, **kwargs):
        self.filename = session_filename
        self.emulate = emulate
        self._set_config(kwargs)
        self._get_corpus()
        self.recorded_precision = []
        self.load_session()
        self.user_features = None
        self.new_instances = 0
        self.new_features = 0
        self._train()
        self._build_feature_boost()

    def _set_config(self, config):
        default_config.update(config)
        for key, value in default_config.items():
            if value is not None:
                setattr(self, key, value)

    def _get_corpus(self):
        self.training_corpus = Corpus()
        self.training_corpus.load_from_file(self.training_corpus_f)

        self.unlabeled_corpus = Corpus()
        self.unlabeled_corpus.load_from_file(self.u_corpus_f)

        self.test_corpus = Corpus()
        self.test_corpus.load_from_file(self.test_corpus_f)

        self.user_corpus = Corpus()

    def _build_feature_boost(self):
        """Creates the user_features array with defaults values."""
        alpha = self.classifier.alpha
        self.n_class, self.n_feat = self.classifier.feature_log_prob_.shape
        self.user_features = array([[alpha] * self.n_feat] * self.n_class)

    def _train(self):
        """Fit the classifier with the training set plus the new vectors and
        features. Then performs a step of EM.
        """
        if len(self.user_corpus):
            self.classifier.fit(
                vstack((self.training_corpus.instances,
                        self.user_corpus.instances), format='csr'),
                (self.training_corpus.primary_targets +
                 self.user_corpus.primary_targets),
                features=self.user_features
            )
        else:
            self.classifier.fit(self.training_corpus.instances,
                                self.training_corpus.primary_targets,
                                features=self.user_features)
        self.recorded_precision.append({
            'testing_presition' : self.evaluate_test(),
            'training_presition' : self.evaluate_training(),
            'new_instances' : self.new_instances,
            'new_features' : self.new_features,
        })
        self.new_instances = 0
        self.new_features = 0
        self.classes = self.classifier.classes_.tolist()

    def _expectation_maximization(self):
        """Performs one cicle of expectation maximization.

        Adds to the training set the em_adding_instances most
        probable instances and removes them from the unlabeled corpus.
        """
        # En este EM convendria agregar solo instancias que no sean de clase "otro"?
        # E-step: Classify the unlabeled pool
        predicted_proba = self.classifier.predict_proba(
            self.unlabeled_corpus.instances
        )
        # Select the k most problable instances
        # en el eje cero estan las clases
        # cual es la clase mas probable para cada instancia
        class_per_instance = array([x[len(self.classes)-1]
                                   for x in predicted_proba.argsort(axis=1)])
        # Probabilidad de la prediccion para la instancia i
        prob_per_instance = predicted_proba.max(axis=1)
        em_k = self.em_adding_instances
        selected_instances = prob_per_instance.argsort()[:em_k]
        # Add instances to training_corpus and remove them from unlabeled_corpus
        selected_instances.sort()
        for i in selected_instances[::-1]:
            self.training_corpus.add_instance(
                *self.unlabeled_corpus.pop_instance(i)
            )

    def predict(self, question):
        return self.classifier.predict(question)

    def instance_bootstrap(self, get_class):
        """Presents a new question to the user until the answer is 'stop'.

        Args:
            get_class: A function that takes the representation of an instance
            and a list of possible classes. Returns the correctclass for the
            instance.
        """
        while True:
            new_index = self.get_next_instance()
            new_instance = self.unlabeled_corpus.instances[new_index]
            representation = self.unlabeled_corpus.representations[new_index]
            if (self.emulate and
                self.unlabeled_corpus.primary_targets[new_index]):
                prediction = self.unlabeled_corpus.primary_targets[new_index][0]
                message = "Adding instance {}, {}".format(representation,
                                                          prediction)
                print colored(message, "red")
            if (not self.emulate or
                not self.unlabeled_corpus.primary_targets[new_index]):
                classes = self._most_probable_classes(new_instance)
                prediction = get_class(representation, classes)
            if prediction == 'stop':
                break
            if prediction == 'train':
                self._train()
                self._expectation_maximization()

            self.new_instances += 1
            self.user_corpus.add_instance(
                *self.unlabeled_corpus.pop_instance(new_index)
            )


    def feature_bootstrap(self, get_class, get_labeled_features):
        """Presents a class and possible features until the prediction is stop.

        Args:
            get_class: A function that receives a list of classes and returns
            one of them. Can return None in case of error.
            get_labeled_features: A function that receives a class and a list
            of features. It must return a list of features associated with the
            class. Can return None in case of error.
        """
        stop = False
        while not stop:
            class_name = get_class(self.get_class_options())
            if not class_name:
                continue
            feature_numbers = self.get_next_features(class_name)
            feature_names = [self.training_corpus.get_feature_name(pos)
                             for pos in feature_numbers]
            prediction = get_labeled_features(class_name, feature_names)
            if not prediction:
                continue
            if prediction == 'stop':
                break
            if prediction == 'train':
                self._train()
                self._expectation_maximization()

            prediction = [feature_names.index(f) for f in prediction]

            for feature in prediction:
                self.user_features[class_number][feature] += \
                    self.feature_boost
                self.new_features += 1


    def _most_probable_classes(self, instance):
        """Return a list of the most probable classes for the given instance.

        Args:
            instance: a vector with the instance to be classified

        Returns:
            A list of classes of len given by the number_of_classes in the
            initial configuration.
        """
        classes = self.classifier.predict_log_proba(instance)
        indexes = classes.argsort()
        result = []
        indexes = indexes[0].tolist()
        indexes.reverse()
        for index in indexes[:self.number_of_classes]:
            result.append(self.classes[index])  # Class name
        result.append(self.classes[-1])
        return result

    def get_next_instance(self):
        """Selects an instance to be sent to the user.

        This is the core of the active learning algorithm.

        Returns:
            The index of an instance selected from the unlabeled_corpus.
        """
        try:
            index = randint(0, len(self.unlabeled_corpus))
            return index
        except IndexError:
            return None

    def get_class_options(self):
        """Sorts a list of classes to present to the user by relevance.

        The user will choose one to label features associated with the class.

        Returns:
            A list of classes.
        """
        return self.classes

    def get_next_features(self, class_name):
        """Selects a  and a list of features to be sent to the oracle.

        Returns:
            A list of features numbers.
        """
        selected_f_pos = self.classifier.feat_information_gain.argsort()[:20]
        return selected_f_pos

    def evaluate_test(self):
        """Evaluates the classifier with the testing set.

        Returns:
            The score of the classifier over the test corpus
        """
        return self.classifier.score(self.test_corpus.instances,
                                     self.test_corpus.primary_targets)

    def evaluate_training(self):
        """Evaluate the accuracy of the classifier with the labeled data.

        Returns:
            The score of the classifier over the training corpus
        """
        # Agregamos la evidencia del usuario para evaluacion?
        return self.classifier.score(self.training_corpus.instances,
                                     self.training_corpus.primary_targets)

    def get_report(self):
        """
        Returns:
            A sklearn.metrics.classification_report on the performance
            of the cassifier over the test corpus.
        """
        predicted_targets = self.predict(self.test_corpus.instances)
        return classification_report(self.test_corpus.primary_targets,
                                     predicted_targets)

    def label_corpus(self):
        """Saves the classes of the unlabeled_corpus in a file.

        The filename must be passed into the configuration under the name
        u_corpus_f.

        Returns:
            The number of new classes added to the corpus.
        """
        self.unlabeled_corpus.concetenate_corpus(self.user_corpus)
        self.unlabeled_corpus.save_to_file(self.u_corpus_f)

    def save_session(self, filename):
        """Saves the instances and targets introduced by the user in filename.

        Writes a pickle tuple in the file that can be recovered using the
        method load_session.

        Returns:
            False in case of error, True in case of success.
        """
        if not filename:
            return False
        if not (self.user_vectors != None or self.user_features != None):
            return False

        f = open(filename, 'w')
        to_save = {'training_corpus': self.training_corpus,
                   'unlabeled_corpus': self.unlabeled_corpus,
                   'user_corpus': self.user_corpus,
                   'user_features': self.user_features,
                   'recorded_precision': self.recorded_precision}
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
        loaded_data = pickle.load(f)
        f.close()
        self.training_corpus = loaded_data['training_corpus']
        self.unlabeled_corpus = loaded_data['unlabeled_corpus']
        self.user_corpus = loaded_data['user_corpus']
        self.user_features = loaded_data['user_features']
        self.recorded_precision = loaded_data['recorded_precision']
        return True
