import unittest
from metrics import LearningCurve


class TestLearningCurve(unittest.TestCase):

    def setUp(self):
        self.metric = LearningCurve()

    def test_empty_session(self):
        self.metric.session = None
        self.metric.get_from_session()
        self.assertEqual(self.metric.info, '')

    def test_full_session(self):
        p1 = {'training_presition': 0.4, 'testing_precision': 0.6,
              'new_features': 4, 'new_instances': 2}
        p2 = {'training_presition': 0.5, 'testing_precision': 0.7,
              'new_features': 2, 'new_instances': 1}
        self.metric.session = {'recorded_precision': [p1, p2]}
        self.metric.get_from_session()
        expected = '6\t0.6\n9\t0.7\n'
        self.assertEqual(expected, self.metric.info)


if __name__ == '__main__':
    unittest.main()