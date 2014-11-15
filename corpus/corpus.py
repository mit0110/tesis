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
        -- extra_info: a dictionary where the values are lists. Each list
        represents extra information about each instance, for example, entropy.
        The default value for each list is 0.

    """
    def __init__(self):
        self.instances = []
        self.representations = []
        self.primary_targets = []
        self.full_targets = []
        self._features_vectorizer = None
        self.extra_info = {}

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
            if mode(target)[1][0] != 1:
                self.primary_targets.append(mode(target)[0][0])
            else:
                self.primary_targets.append(target[0])
        else:
            self.primary_targets.append(None)

        for key in self.extra_info:
            self.extra_info[key].append(0)

    def add_extra_info(self, name, values=None):
        """Appends a field into the extra_info dictionary with values.

        If there is a field with that name in extra info, it will be replaced.

        Args:
            name: a string. It will be the key of the extra_info dictionary.
            values: a list. The len of values must be the same as the len of the
            corpus. If values is not provided, the extra_info field will be
            assigned to a list of 0 of correct len.

        Returns:
            True in case of success, False in case of error.
        """
        if not values:
            self.extra_info[name] = [0] * len(self)
        elif isinstance(values, list) and len(values) == len(self):
            self.extra_info[name] = values
        else:
            return False
        return True

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
        for _, v in self.extra_info.items():
            v.pop(index)
        return (row, target, representation)

    def __len__(self):
        if isinstance(self.instances, list):
            return len(self.instances)
        return self.instances.shape[0]

    def concetenate_corpus(self, new_corpus):
        """Adds all the elements of new_corpus into the current corpus.

        Does not check for duplicated values. The object owns the element of the
        new corpus after this call, consider passing a copy.

        Only will be copied the extra_info fields that both corpus share.

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
        for k in self.extra_info:
            if not k in new_corpus.extra_info:
                self.extra_info[k] += [0] * len(new_corpus)
            else:
                self.extra_info[k] += new_corpus.extra_info[k]

    def check_consistency(self):
        len_components = (self.instances.shape[0] == len(self.full_targets) ==
                          len(self.primary_targets) ==
                          len(self.representations))
        len_extra_info = reduce(lambda x, y: x and y,
                                [len(self.primary_targets) == len(i)
                                 for i in self.extra_info.values()],
                                True)
        return len_components and len_extra_info