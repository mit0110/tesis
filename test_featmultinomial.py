"""
Tests for the modified MultinonialNB that checks if it is correctly trained
using labeled features.
"""

import unittest
import numpy as np

from featmultinomial import FeatMultinomalNB
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



if __name__ == '__main__':
    unittest.main()
