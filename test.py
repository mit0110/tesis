"""
Tests for the modified MultinonialNB that checks if it is correctly trained
using labeled features.
"""

import unittest

from featmultinomial import FeatMultinomalNB
from copy import deepcopy
from math import log

# I should use random numbers here!
X = [
    [0, 1, 0, 2],
    [1, 2, 4, 0],
    [0, 1, 2, 1]
    ]

Y = [1, 0, 0]

features = [
            [1, 1, 1.5, 1],
            [1, 1.5, 1, 1]
           ]

class TestFeatMultinomialNB(unittest.TestCase):

    def test_fit(self):
        fmnb = FeatMultinomalNB()
        fmnb.fit(X, Y)
        no_feat_prior = deepcopy(fmnb.feature_log_prob_)
        fmnb.fit(X, Y, features=features)
        feat_prior = fmnb.feature_log_prob_
        self.assertNotEqual(no_feat_prior[0][2], feat_prior[0][2])
        self.assertEqual(fmnb.alpha, features)
#        self.assertEqual(no_feat_prior[0][2], feat_prior[0][2] - log(0.5))

    def test_information_gain(self):
        fmnb = FeatMultinomalNB()
        fmnb.fit(X, Y)
        n_features = len(X[0])
        ig = fmnb.information_gain()
        self.assertEqual(ig.shape, (n_features, ))
        answer = []


if __name__ == '__main__':
    unittest.main()

