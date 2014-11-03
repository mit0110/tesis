import pickle
import scipy.sparse.csr

from scipy.sparse import vstack, csr_matrix
from scipy.stats import mode


class Corpus(object):
    """A representation of corpus with instances and multiple targets.

    Attributes:
        -- instances: a scipy sparse matrix with the processed instances of the
        corpus. Must be a csr matrix or you need to reimplement the method
        add_instance_target.
        -- primary_targets: a vector with the most important target
        classes for each instance.
        -- full_targets: a vector with the full list of classes associated to
        each instance.
        -- representations: a natural language representation for each of the
        instances.

    """
    def __init__(self):
        self.instances = []
        self.representations = []
        self.primary_targets = []
        self.full_targets = []
        self._features_vectorizer = None

    def load_from_file(self, filename):
        f = open(filename, 'r')
        (self.instances, self.full_targets, self.representations,
            self._features_vectorizer) = pickle.load(f)
        self.calculate_primary_targets()
        f.close()

    def save_to_file(self, filename):
        f = open(filename, 'w')
        pickle.dump((self.instances, self.full_targets,
                     self.representations, self._features_vectorizer), f)
        f.close()

    def calculate_primary_targets(self):
        """Selects the primary target for each instance from self.full_targets.

        The primary target is the one that most occur, or the first one if all
        the elements occur the same number of times.
        """
        self.primary_targets = []
        for targets in self.full_targets:
            if not targets:
                self.primary_targets.append(None)
                continue
            m = mode(targets)
            if m[1][0] != 1:
                self.primary_targets.append(m[0])
            else:
                self.primary_targets.append(targets[0])

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
        if isinstance(self.instances, scipy.sparse.csr.csr_matrix):
            instance = csr_matrix(instance)
            self.instances = vstack((self.instances, instance), format='csr')
        else:
            self.instances = csr_matrix(instance)
        self.full_targets.append(target)
        self.representations.append(representation)
        if target:
            p_target = mode(target)[0] if mode(target)[1][0] != 1 else target[0]
            self.primary_targets.append(p_target)
        else:
            self.primary_targets.append(None)

    def pop_instance(self, index):
        """Deletes the instance and returns a copy.

        Returns:
            A tuple where the first element is the instances, the second
            element is the list of targets and the third element is the
            representation.
        """
        row = self.instances.getrow(index)
        m1 = self.instances[:index]
        m2 = self.instances[index+1:]
        if m1.shape[0] and m2.shape[0]:
            self.instances = vstack((m1, m2), format='csr')
        elif m1.shape[0]:
            self.instances = m1
        else:
            self.instances = m2
        target = self.full_targets.pop(index)
        representation = self.representations.pop(index)
        self.primary_targets.pop(index)
        return (row, target, representation)

    def __len__(self):
        if isinstance(self.instances, list):
            return len(self.instances)
        return self.instances.shape[0]

    def concetenate_corpus(self, new_corpus):
        """Adds all the elements of new_corpus into the current corpus.

        Does not check for duplicated values. The object owns the element of the
        new corpus after this call, consider passing a copy.

        Args:
            new_corpus: an instance of Corpus. It must have the same amount
            of features and be obtained by the same vectorizer object.
        """
        if (isinstance(self.instances, scipy.sparse.csr.csr_matrix)
            and isinstance(new_corpus.instances, scipy.sparse.csr.csr_matrix)):
            self.instances = vstack((self.instances, new_corpus.instances),
                                    format='csr')
        elif isinstance(new_corpus.instances, scipy.sparse.csr.csr_matrix):
            self.instances = new_corpus.instances
        self.full_targets += new_corpus.full_targets
        self.representations += new_corpus.representations
        self.primary_targets += new_corpus.primary_targets

    def check_consistency(self):
        return (self.instances.shape[0] == len(self.full_targets) ==
                len(self.primary_targets) == len(self.representations))