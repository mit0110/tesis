
import unittest
import numpy as np
import mock

from activepipe import ActivePipeline
from featureforge.vectorizer import Vectorizer
from corpus import Corpus
from scipy.sparse import csr_matrix

testing_config = {
    'features': Vectorizer([lambda x : x]),
    'em_adding_instances': 3,
    'u_corpus_f': 'test_files/unlabeled_corpus.pickle',
    'test_corpus_f': 'test_files/test_corpus.pickle',
    'training_corpus_f': 'test_files/training_corpus.pickle',
    'dummy_config' : None,
    'number_of_features': 2,
}

X = [
    [0.0, 1.0, 0.0],
    [1.0, 1.0, 1.0],
    [0.0, 1.0, 1.0]
    ]

Y = [[1], [0, 1] , [0, 0]]

U_vectors = [
    [1.0, 1.0, 1.0],
    [1.0, 1.0, 2.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [2.0, 2.0, 2.0],
]

T_vectors = [
    [1.0, 3.0, 5.0],
    [3.0, 1.0, 0.0]
]

T_targets = [[1], [2]]

def _eq_crs_matrix(m1, m2):
    return not (m1 != m2).todense().any()


class TestActivePipe(unittest.TestCase):

    def setUp(self):
        self.pipe = ActivePipeline(**testing_config)

    def test_em_select(self):
        """Tests if the rigth instances are selected to add in training vectors.
        """
        self.pipe._expectation_maximization()
        self.assertTrue(_eq_crs_matrix(csr_matrix([1.0, 1.0, 1.0]),
                                       self.pipe.training_corpus.instances[-1]))
        self.assertTrue(_eq_crs_matrix(csr_matrix([1.0, 0.0, 0.0]),
                                       self.pipe.training_corpus.instances[-2]))
        self.assertTrue(_eq_crs_matrix(csr_matrix([2.0, 2.0, 2.0]),
                                       self.pipe.training_corpus.instances[-3]))

        self.assertEqual(len(self.pipe.unlabeled_corpus), 2)

    def test_get_corpus(self):
        """Test the three corpus loaded from files."""
        self.assertEqual(len(self.pipe.training_corpus), len(X))
        self.assertEqual(len(self.pipe.test_corpus), len(T_vectors))
        self.assertEqual(len(self.pipe.unlabeled_corpus), len(U_vectors))
        self.assertEqual(len(self.pipe.user_corpus), 0)

    def test_set_config(self):
        """Each configuration must be set as attribute if not None."""
        for key, value in testing_config.items():
            if value is not None:
                self.assertTrue(hasattr(self.pipe, key))
                self.assertEqual(getattr(self.pipe, key), value)
            else:
                self.assertFalse(hasattr(self.pipe, key))

    def test_get_next_features(self):
        """Tests if the features are selected in order of IG for the class."""
        self.pipe.classifier.feat_information_gain = np.array([2, 0, 1])
        feat_indexes = self.pipe.get_next_features(class_number=0)
        self.assertEqual(feat_indexes, [2, 1])

    def test_get_next_features_repeated(self):
        """If the features where labeled, don't ask for them again."""
        self.pipe.classifier.feat_information_gain = np.array([2, 0, 1])
        self.pipe.user_features[0][0] = 5
        self.pipe.user_features[0][2] = 5
        feat_indexes = self.pipe.get_next_features(class_number=0)
        self.assertEqual(feat_indexes, [1])

    def test_handle_feature_prediction(self):
        """Positive and negative examples must be added to user_features"""
        class_number = 0
        self.pipe.handle_feature_prediction(class_number, full_set=[0, 1, 2],
                                            prediction=[1])
        self.assertEqual(self.pipe.user_features[0][1],
                         self.pipe.classifier.alpha + self.pipe.feature_boost,
                         msg='Bad Positive Example')
        self.assertEqual(self.pipe.user_features[0][2],
                         self.pipe.classifier.alpha - self.pipe.feature_boost,
                         msg='Bad negative Example')
        self.assertEqual(self.pipe.user_features[0][0],
                         self.pipe.classifier.alpha - self.pipe.feature_boost,
                         msg='Bad negative Example')
        self.assertTrue(np.all(self.pipe.user_features[1] ==
                               self.pipe.classifier.alpha),
                        msg='Change in non labeled feature')


if __name__ == '__main__':
    unittest.main()
