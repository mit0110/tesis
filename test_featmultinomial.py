"""
Tests for the modified MultinonialNB that checks if it is correctly trained
using labeled features.
"""

import unittest
import numpy as np

from featmultinomial import FeatMultinomalNB
from copy import deepcopy
from math import log
from sklearn import tree


# I should use random numbers here!
X = np.array([
    [2, 0, 10, 3],
    [5, 0, 1, 0],
    [0, 8, 3, 7]
    ])

Y = np.array([0, 0, 1])

features = np.array([
            [1, 1, 1.5, 1],
            [1, 1.5, 1, 1]
           ])

"""
I = [[1,0,1,1],
     [1,0,1,0],
     [0,1,1,1]]

P(I0=1, c0) = #instances with feat 0 and class 0 / # instances = 2/3
P(I1=1, c0) = 0/3 = 0
P(I2=1, c0) = 2/3
P(I3=1, c0) = 1/3

P(I0=1, c1) = #instances with feat 0 and class 1 / # instances = 0/3 = 0
P(I1=1, c1) = 1/3
P(I2=1, c1) = 1/3
P(I3=1, c1) = 1/3

P(I0=0, c0) = #instances without feat 0 and class 0 / # instances = 0/3 = 0
P(I1=0, c0) = 2/3
P(I2=0, c0) = 0/3
P(I3=0, c0) = 1/3

P(I0=0, c1) = #instances with feat 0 and class 1 / # instances = 1/3
P(I1=0, c1) = 0/3
P(I2=0, c1) = 0/3
P(I3=0, c1) = 0/3

P(I0=1) = 2/3   P(I0=0) = 1/3
P(I1=1) = 1/3   P(I1=0) = 2/3
P(I2=1) = 1     P(I2=0) = 0
P(I3=1) = 2/3   P(I3=0) = 1/3

P(c0) = 2/3
P(c1) = 1/3

IG(f0)  = (P(I0=1, c0) * log(P(I0=1, c0) / (P(I0=1) * P(c0)) ) ) +
          (P(I0=1, c1) * log(P(I0=1, c1) / (P(I0=1) * P(c1)) ) ) +
          (P(I0=0, c0) * log(P(I0=0, c0) / (P(I0=0) * P(c0)) ) ) +
          (P(I0=0, c1) * log(P(I0=0, c1) / (P(I0=0) * P(c1)) ) )
        = (2.0/3.0 * log(2.0/3.0 / (2.0/3.0 * 2.0/3.0) ) ) +
          (0 * log(0 / (2.0/3.0 * 1/3.0) ) ) +
          (0 * log(0 / (1/3.0 * 2.0/3.0) ) ) +
          (1/3.0 * log(1/3.0 / (1/3.0 * 1/3.0) ) )
        = 0.27031 + 0 + 0 + 0.3662 = 0.63651

IG(f1)  = (0 * log(0 / (1/3.0 * 2/3.0) ) ) +
          (1.0/3.0 * log(1.0/3.0 / (1.0/3.0 * 1.0/3.0) ) ) +
          (2/3.0 * log(2/3.0 / (2.0/3.0 * 2/3.0) ) ) +
          (0 * log(1.0/3.0 / (2.0/3.0 * 1.0/3.0) ) )
        = 0 + 0.3662 + 0.27031 + 0.13515 = 0.7716

IG(f2)  = (2/3.0 * log(2/3.0 / (1 * 2/3.0) ) ) +
          (1/3.0 * log(1/3.0 / (1 * 1/3.0) ) ) +
          (0 * log(0 / (0 * 2/3.0) ) ) +
          (0 * log(0 / (0 * 1/3.0) ) )
        = 0 + 0 + 0 + 0 = 0


IG(f3)  = (1/3.0 * log(1/3.0 / (2/3.0 * 2/3.0) ) ) +
          (1/3.0 * log(1/3.0 / (2/3.0 * 1/3.0) ) ) +
          (1/3.0 * log(1/3.0 / (1/3.0 * 2/3.0) ) ) +
          (0 * log(0 / (1/3.0 * 1/3.0) ) )
        = -0.09589402415059363 + 0.135115 + 0.135115 + 0 = 0.17433

"""
ig_correct_anwers = [0.636514, 0.636514, 0.0, 0.17441]

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
        self.assertEqual(ig.shape[0], X.shape[1])
        for i, answer in enumerate(ig):
            self.assertAlmostEqual(answer, ig_correct_anwers[i], places=4)
        self.assertTrue(np.all(ig.argsort() == [2, 3, 0, 1]))


class TestIGwithDecisionTree(unittest.TestCase):
    def setUp(self):
      self.fmnb = FeatMultinomalNB()
      self.dtree = tree.DecisionTreeClassifier(criterion='entropy',
                                               min_samples_split=1,
                                               min_samples_leaf=1)

    def tearDown(self):
      self.assertTrue(np.all(self.fmnb.feat_information_gain.argsort() ==
                             self.dtree.feature_importances_.argsort()))

    def test_ig_with_iris(self):
      from sklearn.datasets import load_iris
      iris = load_iris()
      self.fmnb.fit((iris.data > 3), iris.target)
      self.dtree.fit((iris.data > 3), iris.target)

    def test_ig_with_bag_of_words(self):
      from sklearn.feature_extraction.text import CountVectorizer
      corpus = ['This is a corpus and the main',
                'objective is to have senteces to simulate a sparse',
                'matrix of features, on the opposite of the iris corpus',
                'that has few features and all the features are present in all',
                'the instances.',
                'By the way, this all are documents.']
      target = [1, 2, 3, 2, 1, 2]
      vectorizer = CountVectorizer(min_df=1)
      X = vectorizer.fit_transform(corpus)
      self.fmnb.fit(X, target)
      self.dtree.fit(X.todense(), target)


if __name__ == '__main__':
    unittest.main()
