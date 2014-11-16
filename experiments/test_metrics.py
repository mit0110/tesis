import unittest
from metrics import LearningCurve, PrecisionRecall


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


if __name__ == '__main__':
    unittest.main()