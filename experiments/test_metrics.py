import unittest
import numpy as np

from metrics import (LearningCurve, PrecisionRecall, KappaStatistic,
                     pr_from_confusion_matrix)


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


class TestPrecisionRecall(TestLearningCurve):

    def setUp(self):
        self.metric = PrecisionRecall()

    def test_full_session(self):
        fake_report = ('blank\ntitles\nblank\nclass1\tp1\tp2\tp3\tp4\n'
                       'class2\tr1\tr2\tr3\tr4\n'
                       'class3\tq1\tq2\tq3\tq4\n'
                       'blank\ntotal\nblank\n')
        self.metric.session = {'classification_report': fake_report}
        self.metric.get_from_session()
        expected = ('class1\tp1\tp2\tp3\n'
                    'class2\tr1\tr2\tr3\n'
                    'class3\tq1\tq2\tq3\n')
        self.assertEqual(expected, self.metric.info)


class TestKappaStatistic(TestLearningCurve):

    def setUp(self):
        self.metric = KappaStatistic()

    def test_full_session(self):
        # http://stats.stackexchange.com/questions/82162/kappa-statistic-in-plain-english
        fake_confusion_matrix = np.array([[22, 9], [7, 13]])
        expected = 0.3534
        self.metric.session = {'recorded_precision': [{'confusion_matrix':
                                                       fake_confusion_matrix}]}
        self.metric.get_from_session()
        self.assertAlmostEqual(expected, float(self.metric.info), 3)

    def test_full_session2(self):
        # http://en.wikipedia.org/wiki/Cohen%27s_kappa
        fake_confusion_matrix = np.array([[20, 5], [10, 15]])
        expected = 0.39999999
        self.metric.session = {'recorded_precision': [{'confusion_matrix':
                                                       fake_confusion_matrix}]}
        self.metric.get_from_session()
        self.assertAlmostEqual(expected, float(self.metric.info), 3)


class TestPrecisionRecallCM(unittest.TestCase):

    def test_from_cm(self):
        """Test the values of precision and recall from confusion matrix"""
        fake_confusion_matrix = np.array([[22, 9], [7, 13]])
        expected = [(0.709677,0.75862069), (0.590909091, 0.65)]
        result = pr_from_confusion_matrix(fake_confusion_matrix)
        self.assertEqual(expected, result)


if __name__ == '__main__':
    unittest.main()