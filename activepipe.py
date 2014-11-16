import pickle
import numpy as np

from corpus import Corpus
from random import randint
from scipy.sparse import vstack
from sklearn.metrics import precision_score, classification_report
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.preprocessing import normalize
from termcolor import colored

from defaults import default_config


class ActivePipeline(object):
    """
    Attributes:
        session_filename:

        emulate: A boolean. If is set to True, the pipe will search for labels
        in the unlabeled_corpus and in the feature_corpus and will only ask the
        user if there is no information available.

        training_corpus:

        unlabeled_corpus:

        test_corpus:

        feature_corpus: A matrix of shape [n_class, n_feat] with three possible
        values. -1 indicates that the feature was never asked to the user for
        that class, 0 indicates no relation, and 1 indicates relation between
        feature and class. The feature corpus will be loaded from the file
        self.feature_label_f intruduced by the config, and will be used only
        during user emulation. It can be updated using the function
        label_feature_corpus.

        recorded_precision:

        new_instances:

        new_features:

        classes:

        user_features:

        user_corpus:
    """

    def __init__(self, session_filename='', emulate=False, **kwargs):
        """
        Args:
            session_filename: Optional. The name of a file storing a session
            that will be loaded using the method load_session.
            emulate: a boolean. Will set the attribute emulate acordinly.
            **kwargs: the configuration for the pipe. Each parameters passed
            will be converted to an attribute of the pipe. The minimum
            configuration possible is set in the defaults file, and each value
            not passed as a parameter will be taken from there.
        """
        self.session_filename = session_filename
        self.emulate = emulate
        self._set_config(kwargs)
        self._get_corpus()
        self._get_feature_corpus()
        self.recorded_precision = []
        self.load_session()
        self.user_features = None
        self.new_instances = 0
        self.new_features = 0
        self.classes = []
        self._train()
        self._build_feature_boost()

    def _set_config(self, config):
        """Sets the keys of config+default_config dict as an attribute of self.
        """
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

    def _get_feature_corpus(self):
        """Loads the feature corpus from self.feature_corpus_f"""
        f = open(self.feature_corpus_f, 'r')
        self.feature_corpus = pickle.load(f)
        f.close()

    def _build_feature_boost(self):
        """Creates the user_features np.array with defaults values."""
        alpha = self.classifier.alpha
        self.n_class, self.n_feat = self.classifier.feature_log_prob_.shape
        self.user_features = np.array([[alpha] * self.n_feat] * self.n_class)
        if self.emulate:
            self.asked_features = self.feature_corpus != -1
        else:
            self.asked_features = self.user_features != alpha # False

    def _train(self):
        """Fit the classifier with the training set plus the new vectors and
        features. Then performs a step of EM.
        """
        try:
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
        except ValueError:
            import ipdb; ipdb.set_trace()
        self.recorded_precision.append({
            'testing_precision' : self.evaluate_test(),
            'training_precision' : self.evaluate_training(),
            'new_instances' : self.new_instances,
            'new_features' : self.new_features,
        })
        self.new_instances = 0
        self.new_features = 0
        self.classes = self.classifier.classes_.tolist()
        self._retrained = True

    def _expectation_maximization(self):
        """Performs one cicle of expectation maximization.

        Re estimates the parameters of the multinomial (class_prior and
        feature_log_prob_) to maximize the expected likelihood. The likelihood
        is calculated with a probabilistic labeling of the unlabeled corpus
        plus the known labels from the labeled corpus.
        """
        # E-step: Classify the unlabeled pool
        predicted_proba = self.classifier.predict_proba(
            self.unlabeled_corpus.instances
        )
        # M-step: Maximizing the likelihood
        # Unlabeled component
        instance_proba = self.classifier.instance_proba(
            self.unlabeled_corpus.instances
        )
        predicted_proba = predicted_proba.T * instance_proba
        class_prior = predicted_proba.sum(axis=1)
        feature_prob = safe_sparse_dot(predicted_proba,
                                      self.unlabeled_corpus.instances)

        if len(self.training_corpus) != 0:
            # Labeled component
            instance_proba = self.classifier.instance_proba(
                self.training_corpus.instances
            )
            instance_class_matrix = self._get_instance_class_matrix()
            predicted_proba = instance_class_matrix.T * instance_proba
            l_class_prior = predicted_proba.sum(axis=1)
            l_feat_prob = safe_sparse_dot(predicted_proba,
                                          self.training_corpus.instances)
            class_prior = 0.1 * class_prior + 0.9 * l_class_prior
            feature_prob = 0.1 * feature_prob + 0.9 * l_feat_prob

        self.classifier.class_log_prior_ = np.log(class_prior /
                                                  class_prior.sum())
        self.classifier.feature_log_prob_ = np.log(normalize(feature_prob,
                                                             norm='l1'))

    def _get_instance_class_matrix(self):
        """Returns a binary matrix for the training instances and its labels.

        Returns:
            An array like, shape = [n_instances, n_class]. Each element is
            one if the instances is labeled with the class in the training
            corpus.
        """
        m1 = np.arange(len(self.classes))
        m1 = m1.reshape((1, len(self.classes)))
        m1 = np.repeat(m1, len(self.training_corpus), axis=0)
        m2 = np.zeros((len(self.training_corpus), len(self.classes)))
        for i in range(len(self.training_corpus)):
            class_index = self.classes.index(
                self.training_corpus.primary_targets[i]
            )
            m2[i] = class_index
        result = (m1 == m2).astype(np.int8, copy=False)
        assert np.all(result.sum(axis=1) == 1)
        assert result.sum() == len(self.training_corpus)
        return result

    def predict(self, question):
        return self.classifier.predict(question)

    def instance_bootstrap(self, get_labeled_instance, max_iterations=None):
        """Presents a new question to the user until the answer is 'stop'.

        Args:
            get_labeled_instance: A function that takes the representation of
            an instance and a list of possible classes. Returns the correct
            class for the instance.
            max_iterations: Optional. An interger. The cicle will execute at
            most max_iterations times if the user does not enter stop before.

        Returns:
            The number of instances the user has labeled.
        """
        it = 0
        result = 0
        while not max_iterations or it < max_iterations:
            it += 1
            new_index = self.get_next_instance()
            new_instance = self.unlabeled_corpus.instances[new_index]
            representation = self.unlabeled_corpus.representations[new_index]
            if (self.emulate and
                self.unlabeled_corpus.primary_targets[new_index]):
                prediction = self.unlabeled_corpus.primary_targets[new_index]
                message = "Emulation: Adding instance {}, {}".format(
                    representation, prediction
                )
                print colored(message, "red")
            if (not self.emulate or
                not self.unlabeled_corpus.primary_targets[new_index]):
                classes = self._most_probable_classes(new_instance)
                prediction = get_labeled_instance(representation, classes)
            if prediction == 'stop':
                break
            if prediction == 'train':
                self._train()
                self._expectation_maximization()
                continue

            self.new_instances += 1
            result += 1
            instance, targets, r = self.unlabeled_corpus.pop_instance(new_index)
            self.user_corpus.add_instance(
                instance, [prediction] + targets, r
            )

        return result

    def feature_bootstrap(self, get_class, get_labeled_features,
                          max_iterations=None):
        """Presents a class and possible features until the prediction is stop.

        Args:
            get_class: A function that receives a list of classes and returns
            one of them. Can return None in case of error.
            get_labeled_features: A function that receives a class and a list
            of features. It must return a list of features associated with the
            class. Can return None in case of error.
            max_iterations: Optional. An interger. The cicle will execute at
            most max_iterations times if the user does not enter stop before.

        Returns:
            The number of features the user has labeled.
        """
        it = 0
        result = 0
        while not max_iterations or it < max_iterations:
            it += 1
            class_name = get_class(self.get_class_options())
            if not class_name:
                continue
            if class_name == 'stop':
                break
            if class_name == 'train':
                self._train()
                self._expectation_maximization()
                continue
            class_number = self.classes.index(class_name)
            feature_numbers = self.get_next_features(class_number)
            e_prediction = []
            prediction = []
            if self.emulate:
                e_prediction = [f for f in feature_numbers
                                if self.feature_corpus[class_number][f] == 1]
                feature_numbers = [f for f in feature_numbers
                                   if f not in e_prediction]
                print "Adding {} features from corpus".format(len(e_prediction))
            if feature_numbers:
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
                    continue
                prediction = [feature_names.index(f) for f in prediction]
            self.handle_feature_prediction(class_number,
                                           feature_numbers + e_prediction,
                                           prediction + e_prediction)
            result += len(prediction + e_prediction)
        return result

    def handle_feature_prediction(self, class_number, full_set, prediction):
        """Adds the new information from prediction to user_features.

        Args:
            class_number: an interger. The position of the class in self.classes
            full_set: a list of positions of features that was given to the
            user.
            prediction: a list of positions of features selected for the class.
            The features not present in this class are considered as negative
            examples.
        """
        for feature in full_set:
            if feature in prediction:
                self.user_features[class_number][feature] += \
                     self.feature_boost
            self.asked_features[class_number][feature] = True
        self.new_features += len(prediction)

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
            result.append(self.classes[index])
        result.append(self.classes[-1])
        return result

    def get_next_instance(self):
        """Selects the index of an unlabeled instance to be sent to the user.

        Returns:
            The index of an instance selected from the unlabeled_corpus.
        """
        if len(self.unlabeled_corpus) == 0:
            return None
        if self._retrained:
            self.u_clasifications = self.classifier.predict_proba(
                self.unlabeled_corpus.instances
            )
            entropy = self.u_clasifications * np.log(self.u_clasifications)
            entropy = entropy.sum(axis=1)
            entropy *= -1
            self.unlabeled_corpus.add_extra_info('entropy', entropy.tolist())

            self._retrained = False
        # Select the instance
        min_entropy = min(self.unlabeled_corpus.extra_info['entropy'])
        return self.unlabeled_corpus.extra_info['entropy'].index(min_entropy)

    def get_class_options(self):
        """Sorts a list of classes to present to the user by relevance.

        The user will choose one to label features associated with the class.

        Returns:
            A list of classes.
        """
        return self.classes

    def get_next_features(self, class_number):
        """Selects a  and a list of features to be sent to the oracle.

        Args:
            class_number: An interger. The position of the class where the
            features will belong in the np.array self.classes.

        Returns:
            A list of features numbers of size self.number_of_features.
        """
        # Select the positions of the features that cooccur most with the class
        selected_f_pos = self.classifier.feature_count_[class_number].argsort()
        # Eliminate labeled features
        def non_seen_filter(i):
            return not self.asked_features[class_number][i]
        selected_f_pos = filter(non_seen_filter, selected_f_pos.tolist())

        selected_f_pos = selected_f_pos[:-(self.number_of_features+1):-1]
        # Sort the features by IG
        def key_fun(i): return -1*self.classifier.feat_information_gain[i]
        selected_f_pos.sort(key=key_fun)
        return selected_f_pos
        # selected_f_pos = self.classifier.feat_information_gain.argsort()[:-100:-1]
        # coocurrence_with_class = []
        # for feat_pos in selected_f_pos:
        #     coocurrence_with_class.append(
        #         self.classifier.feature_count_[class_number][feat_pos]
        #     )
        # coocurrence_with_class = np.array(coocurrence_with_class)
        # coocurrences_order = coocurrence_with_class.argsort()
        # res = [selected_f_pos[i] for i in coocurrences_order[::-1]
        #        if self.user_features[class_number][selected_f_pos[i]] ==
        #           self.classifier.alpha]
        # return np.array(res[:self.number_of_features])

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
        """Adds the user corpus to the unlabeled_corpus and saves it in a file.

        The filename must be passed into the configuration under the name
        u_corpus_f.
        """
        self.unlabeled_corpus.concetenate_corpus(self.user_corpus)
        self.unlabeled_corpus.save_to_file(self.u_corpus_f)

    def label_feature_corpus(self):
        """Adds user_features and asked_features in feature_corpus and saves it.

        The filename must be passed into the configuration under the name
        feature_corpus_f.
        """
        self.feature_corpus = np.where(self.asked_features,
                                       np.zeros((self.n_class, self.n_feat)),
                                       self.feature_corpus)
        self.feature_corpus = np.where(
            self.user_features > self.classifier.alpha,
            np.ones((self.n_class, self.n_feat)),
            self.feature_corpus
        )
        f = open(self.feature_corpus_f, 'w')
        pickle.dump(self.feature_corpus, f)
        f.close()

    def save_session(self, filename):
        """Saves the instances and targets introduced by the user in filename.

        Writes a pickle tuple in the file that can be recovered using the
        method load_session.

        Returns:
            False in case of error, True in case of success.
        """
        if not filename:
            return False
        if not (len(self.user_corpus) != None or self.user_features != None):
            return False

        f = open(filename, 'w')
        to_save = {'training_corpus': self.training_corpus,
                   'unlabeled_corpus': self.unlabeled_corpus,
                   'user_corpus': self.user_corpus,
                   'user_features': self.user_features,
                   'recorded_precision': self.recorded_precision,
                   'asked_features': self.asked_features,
                   'classification_report': self.get_report()}
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
        if not self.session_filename:
            return False
        f = open(self.session_filename, 'r')
        loaded_data = pickle.load(f)
        f.close()
        self.training_corpus = loaded_data['training_corpus']
        self.unlabeled_corpus = loaded_data['unlabeled_corpus']
        self.user_corpus = loaded_data['user_corpus']
        self.user_features = loaded_data['user_features']
        self.recorded_precision = loaded_data['recorded_precision']
        self.asked_features = loaded_data['asked_features']
        return True

