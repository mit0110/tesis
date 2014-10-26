"""
Tests for the modified MultinonialNB that checks if it is correctly trained
using labeled features.
"""

import unittest
import numpy as np
import mock

from activepipe import ActivePipeline
from featmultinomial import FeatMultinomalNB
from featureforge.vectorizer import Vectorizer
from copy import deepcopy
from math import log


# I should use random numbers here!
X = np.array([
    [0.0, 1.0, 0.0],
    [1.0, 1.0, 1.0],
    [0.0, 1.0, 1.0]
    ])

Y = np.array([1, 0, 0])

features = np.array([
            [1, 1, 1.5],
            [1, 1.5, 1]
           ])

class TestFeatMultinomialNB(unittest.TestCase):

    def setUp(self):
        self.fmnb = FeatMultinomalNB()
        self.fmnb.fit(X, Y)

    def test_fit(self):
        no_feat_prior = deepcopy(self.fmnb.feature_log_prob_)
        self.fmnb.fit(X, Y, features=features)
        feat_prior = self.fmnb.feature_log_prob_
        self.assertNotEqual(no_feat_prior[0][2], feat_prior[0][2])
        self.assertTrue(np.all(self.fmnb.alpha == features))
#        self.assertEqual(no_feat_prior[0][2], feat_prior[0][2] - log(0.5))

    def test_information_gain(self):
        ig = self.fmnb.feat_information_gain
        self.assertEqual(ig.shape[0], X.shape[0])
        self.assertTrue(np.all(ig.argsort() == [1, 0, 2]))


testing_config = {
    'features': Vectorizer([lambda x : x]),
    'em_adding_instances': 3,
}


class TestActivePipe(unittest.TestCase):

    def setUp(self):
        def mock_get_corpus_fun(self):
            self.training_vectors = X.tolist()
            self.training_target = Y.tolist()
        self.mock_get_corpus = mock.patch(
            'activepipe.ActivePipeline._get_corpus',
            mock_get_corpus_fun
        )
        self.mock_get_corpus.start()
        self.pipe = ActivePipeline(**testing_config)

    def tearDown(self):
        self.mock_get_corpus.stop()

    def test_em_select(self):
        """Tests if the rigth instances are selected to add in training vectors.
        """
        new_vectors = [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 2.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [2.0, 2.0, 2.0],
        ]
        self.pipe.unlabeled_corpus = [{'question' : i} for i in new_vectors]
        self.pipe._expectation_maximization()
        self.assertIn([1.0, 1.0, 1.0], self.pipe.training_vectors)
        self.assertIn([1.0, 0.0, 0.0], self.pipe.training_vectors)
        self.assertIn([2.0, 2.0, 2.0], self.pipe.training_vectors)

        self.assertEqual(len(self.pipe.unlabeled_corpus), 2)
        self.assertNotIn([1.0, 1.0, 1.0], self.pipe.unlabeled_corpus)
        self.assertNotIn([1.0, 0.0, 0.0], self.pipe.unlabeled_corpus)
        self.assertNotIn([2.0, 2.0, 2.0], self.pipe.unlabeled_corpus)




if __name__ == '__main__':
    unittest.main()

