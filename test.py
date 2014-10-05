"""
Tests for the modified MultinonialNB that checks if it is correctly trained
using labeled features.
"""

import unittest

from featmultinomial import FeatMultinomalNB
from copy import deepcopy

# I should use random numbers here!
X = [
    [0, 1, 0, 2],
    [1, 2, 4, 0],
    [0, 1, 2, 1]
    ]

Y = [1, 0, 0]


class TestFeatMultinomialNB(unittest.TestCase):

    def test_fit(self):
        fmnb = FeatMultinomalNB()
        fmnb.fit(X, Y)
        no_feat_prior = deepcopy(fmnb.feature_log_prob_)
        print no_feat_prior

if __name__ == '__main__':
    unittest.main()

