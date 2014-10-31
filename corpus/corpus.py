import pickle

from scipy.sparse import vstack, csr_matrix


class Corpus(object):
    """A representation of corpus with instances and targets.

    Attributes:
        -- instances: a scipy sparse matrix with the processed instances of the
        corpus. Must be a csr matrix or you need to reimplement the method
        add_instance_target.
        -- targets: a sklearn vector with the target classes for each instance.
        -- representations: a natural language representation for each of the
        instances.

    """
    def __init__(self):
        self.instances = None
        self.representations = None
        self.targets = None
        self._features_vectorizer = None

    def load_from_file(self, filename):
        f = open(filename, 'r')
        (self.instances, self.targets, self.representations,
            self._features_vectorizer) = pickle.load(f)

    def save_to_file(self, filename):
        f = open(filename, 'w')
        pickle.dump((self.instances, self.targets,
                     self.representations, self._features_vectorizer), f)

    def get_feature_name(self, feat_index):
        """Gives the natural language representation of a feature.

        Args:
            feat_index: a non negative interger less than instances.shape[1]

        Returns:
            The name of the feature represented by index feat_index.
        """
        return self._features_vectorizer.column_to_feature(feat_index)[1]

    def add_instance(self, instance, target, representation=None):
        """Adds the given instance, target and representation to the corpus.

        Args:
            instance: a vector with shape equals to self.instances.shape[1]
            target: a list of string representing the classes.
            representation: a string
        """
        instance = csr_matrix(instance)
        self.instances = vstack((self.instances, instance), format='csr')
        self.targets.append(target)
        self.representations.append(representation)

    def pop_instance(self, index):
        """Deletes the instance and returns a copy.

        Returns:
            A tuple where the first element is the instances and the second
            element is the list of targets and the third element is the
            representation.
        """
        row = self.instances.getrow(index)
        m1 = self.instances[:index]
        m2 = self.instances[index+1:]
        self.instances = vstack((m1, m2), format='csr')
        target = self.targets.pop(index)
        representation = self.representations.pop(index)
        return (row, target)

    def __len__(self):
        return len(self.representations)

    def concetenate_corpus(self, new_corpus):
        """Adds all the elements of new_corpus into the current corpus.

        Does not check for duplicated values.

        Args:
            new_corpus: an instance of Corpus. It must have the same amount
            of features and be obtained by the same vectorizer object.
        """
        self.instances = vstack((self.instances, new_corpus.instances),
                                format='csr')
        self.targets += new_corpus.targets
        self.representations += new_corpus.representations