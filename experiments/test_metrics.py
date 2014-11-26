import unittest
import numpy as np

from metrics import (LearningCurve, PrecisionRecall, KappaStatistic,
                     pr_from_confusion_matrix, PrecisionRecallCurve,
                     RecognitionCurve)


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
        expected = [(0.75862069, 0.709677), (0.590909091, 0.65)]
        result = pr_from_confusion_matrix(fake_confusion_matrix)
        for count, (x, y) in enumerate(result):
            self.assertAlmostEqual(expected[count][0], x, 6)
            self.assertAlmostEqual(expected[count][1], y, 6)


    def test_from_big_cm(self):
        """
             TP  FP  FN
        ---------------
        C1 | 22  11  16
        C2 | 13  23  11
        C3 | 17   7  14
        C4 | 24  11  11
        P(c1) = 22/(22+11) = 0.666666666667
        P(c2) = 13/(13+23) = 0.361111111111
        P(c3) = 17/(17+7) = 0.708333333333
        P(c4) = 24/(24+11) = 0.685714285714
        R(c1) = 22/(22+16) = 0.578947368421
        R(c2) = 13/(13+11) = 0.541666666667
        R(c3) = 17/(17+14) = 0.548387096774
        R(c4) = 24/(24+11) = 0.685714285714
        """
        fake_confusion_matrix = np.array([[22,  9,  5,  2],
                                          [ 7, 13,  0,  4],
                                          [ 3,  6, 17,  5],
                                          [ 1,  8,  2, 24]])
        expected = [(0.6666666666666666, 0.5789473684210527),
                    (0.3611111111111111, 0.5416666666666666),
                    (0.7083333333333334, 0.5483870967741935),
                    (0.6857142857142857, 0.6857142857142857)]
        result = pr_from_confusion_matrix(fake_confusion_matrix)
        for count, (x, y) in enumerate(result):
            self.assertAlmostEqual(expected[count][0], x)
            self.assertAlmostEqual(expected[count][1], y)


class TestPrecisionRecallCurve(TestLearningCurve):

    def setUp(self):
        self.metric = PrecisionRecallCurve()

    def test_full_session(self):
        fake_confusion_matrix1 = np.array([[22, 9], [7, 13]])
        fake_confusion_matrix2 = np.array([[20, 3], [5, 15]])
        recorded_precision1 = {'new_instances': 3,
                               'new_features': 4,
                               'confusion_matrix': fake_confusion_matrix1}
        recorded_precision2 = {'new_instances': 1,
                               'new_features': 0,
                               'confusion_matrix': fake_confusion_matrix2}
        self.metric.session = {'classes': ['a', 'b'],
                               'recorded_precision': [recorded_precision1,
                                                      recorded_precision2]}
        expected = ('a\t7\t0.758620689655\t0.709677419355\n'
                    'b\t7\t0.590909090909\t0.65\n'
                    'a\t8\t0.8\t0.869565217391\n'
                    'b\t8\t0.833333333333\t0.75')
        self.metric.get_from_session()

        self.assertEqual(expected, self.metric.info)


class TestRecognitionCurve(TestLearningCurve):

    def setUp(self):
        self.metric = RecognitionCurve()

    def test_full_session(self):
        fake_confusion_matrix1 = np.array([[22, 9, 5], [7, 13, 10], [4, 2, 15]])
        fake_confusion_matrix2 = np.array([[20, 3, 7], [5, 15, 1], [8, 1, 12]])
        recorded_precision1 = {'new_instances': 3,
                               'new_features': 4,
                               'confusion_matrix': fake_confusion_matrix1}
        recorded_precision2 = {'new_instances': 1,
                               'new_features': 0,
                               'confusion_matrix': fake_confusion_matrix2}
        self.metric.session = {'classes': ['a', 'other', 'b'],
                               'recorded_precision': [recorded_precision1,
                                                      recorded_precision2],
                              }
        expected = ('7\t0.649122807018\n'
                    '8\t0.627450980392')
        self.metric.get_from_session()

        self.assertEqual(expected, self.metric.info)


if __name__ == '__main__':
    unittest.main()
