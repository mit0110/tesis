import unittest

from featureforge.validate import BaseFeatureFixture, EQ, IN, APPROX, RAISES

class TestUnigrams(unittest.TestCase, BaseFeatureFixture):
	feature = unigrams