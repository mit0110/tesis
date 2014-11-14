
import unittest
import numpy as np
import mock

from activepipe import ActivePipeline
from corpus import Corpus
from featureforge.vectorizer import Vectorizer
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

testing_config = {
    'features': Vectorizer([lambda x : x]),
    'em_adding_instances': 3,
    'u_corpus_f': 'test_files/unlabeled_corpus.pickle',
    'test_corpus_f': 'test_files/test_corpus.pickle',
    'training_corpus_f': 'test_files/training_corpus.pickle',
    'dummy_config' : None,
    'number_of_features': 2,
}

X = [
    [0.0, 1.0, 0.0],
    [1.0, 1.0, 1.0],
    [0.0, 1.0, 1.0]
    ]

Y = [[1], [0, 1] , [0, 0]]

U_vectors = [
    [1.0, 1.0, 1.0],
    [1.0, 1.0, 2.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [2.0, 2.0, 2.0],
]

T_vectors = [
    [1.0, 3.0, 5.0],
    [3.0, 1.0, 0.0]
]

T_targets = [[1], [2]]


class TestActivePipe(unittest.TestCase):

    def setUp(self):
        self.pipe = ActivePipeline(**testing_config)
        self.instance_class_prob = np.array(
            [[0.5, 0.5],
             [0.25, 0.75],
             [0.7, 0.3],
             [0.1, 0.9],
             [0.8, 0.2]]
        )
        self.instance_prob = np.array([0.02, 0.09, 0.01, 0.12, 0.08])

    def test_em_feat_class_no_labeled(self):
        """Tests if the feature_log_prob matrix is calculated correctly.
        P(fj|ck) = sum_i(P(xi) * fj(xi) * P(ck|xi))
        P(f0|c0) = 0.5*0.02*1 + 0.25*0.09*1 + 0.7*0*0.01 + 0.1*1*0.12 + 0.8*2*0.08 = 0.1725
        P(f0|c1) = 0.5*0.02*1 + 0.75*0.09*1 + 0.3*0*0.01 + 0.9*1*0.12 + 0.2*2*0.08 = 0.2175
        P(f1|c0) = 0.5*0.02*1 + 0.25*0.09*1 + 0.7*0*0.01 + 0.1*0*0.12 + 0.8*2*0.08 = 0.1605
        P(f1|c1) = 0.5*0.02*1 + 0.75*0.09*1 + 0.3*0*0.01 + 0.9*0*0.12 + 0.2*2*0.08 = 0.1095
        P(f2|c0) = 0.5*0.02*1 + 0.25*0.09*2 + 0.7*1*0.01 + 0.1*0*0.12 + 0.8*2*0.08 = 0.19
        P(f2|c1) = 0.5*0.02*1 + 0.75*0.09*2 + 0.3*1*0.01 + 0.9*0*0.12 + 0.2*2*0.08 = 0.18
        """
        expected = np.array([[0.32982792, 0.30688337, 0.36328872],
                             [0.42899408, 0.21597633, 0.35502959]])
        with mock.patch('featmultinomial.FeatMultinomalNB.predict_proba',
                        return_value=self.instance_class_prob) as mock_pred:
            with mock.patch('featmultinomial.FeatMultinomalNB.instance_proba',
                            return_value=self.instance_prob) as mock_inst_p:
                self.pipe.training_corpus = Corpus()
                self.pipe._expectation_maximization()
                np.testing.assert_array_almost_equal(
                    self.pipe.classifier.feature_log_prob_,
                    np.log(expected)
                )

    def test_em_class_no_labeled(self):
        """Tests if the class_log_prior_ matrix is calculated correctly.
        P(ck) = sum_i(P(xi) * P(ck|xi))
        P(c0) = 0.5*0.02 + 0.25*0.09 + 0.7*0.01 + 0.1*0.12 + 0.8*0.08 = 0.1155
        P(c1) = 0.5*0.02 + 0.75*0.09 + 0.3*0.01 + 0.9*0.12 + 0.2*0.08 = 0.2045
        """
        expected = np.array([0.3609375, 0.6390625])

        with mock.patch('featmultinomial.FeatMultinomalNB.predict_proba',
                        return_value=self.instance_class_prob) as mock_pred:
            with mock.patch('featmultinomial.FeatMultinomalNB.instance_proba',
                            return_value=self.instance_prob) as mock_inst_p:
                self.pipe.training_corpus = Corpus()
                self.pipe._expectation_maximization()
                np.testing.assert_array_almost_equal(
                    self.pipe.classifier.class_log_prior_,
                    np.log(expected)
                )

    def test_em_feat_class(self):
        """
        P(fj|ck) = Pu(fj|ck) * 0.1 + 0.9* sum_i(P(xl_i) * fj(xl_i) * {0,1})
        P(f0|c0) = 0.1 * 0.1725 + 0.9 * (0.02*0*0 + 0.09*1*1 + 0.01*0*1) = 0.09825
        P(f0|c1) = 0.1 * 0.2175 + 0.9 * (0.02*0*1 + 0.09*1*0 + 0.01*0*0) = 0.02175
        P(f1|c0) = 0.1 * 0.1605 + 0.9 * (0.02*1*0 + 0.09*1*1 + 0.01*1*1) = 0.10605
        P(f1|c1) = 0.1 * 0.1095 + 0.9 * (0.02*1*1 + 0.09*1*0 + 0.01*1*0) = 0.02895
        P(f2|c0) = 0.1 * 0.19 + 0.9 * (0.02*0*0 + 0.09*1*1 + 0.01*1*1) = 0.109
        P(f2|c1) = 0.1 * 0.18 + 0.9 * (0.02*0*1 + 0.09*1*0 + 0.01*1*0) = 0.018
        """
        expected = np.array([[0.31359719, 0.3384934, 0.34790935],
                             [0.31659388, 0.421397379, 0.262008733]])
        instance_prob_fun = lambda s, x: self.instance_prob[:x.shape[0]]
        with mock.patch('featmultinomial.FeatMultinomalNB.predict_proba',
                        return_value=self.instance_class_prob) as mock_pred:
            with mock.patch('featmultinomial.FeatMultinomalNB.instance_proba',
                            new=instance_prob_fun) as mock_inst_p:
                self.pipe._expectation_maximization()
                np.testing.assert_array_almost_equal(
                    self.pipe.classifier.feature_log_prob_,
                    np.log(expected)
                )

    def test_em_class(self):
        """Tests if the class_log_prior_ matrix is calculated correctly.
        P(ck) = sum_i(P(xui) * P(ck|xui)) * 0.1 + sum_i(P(xli) * P(ck|xli)) * 0.9
        P(c0) = 0.1155 * 0.1 + 0.9 * (0*0.02 + 1*0.09 + 1*0.01) = 0.10155
        P(c1) = 0.2045 * 0.1 + 0.9 * (1*0.02 + 0*0.09 + 0*0.01) = 0.03845
        """
        expected = np.array([0.725357142, 0.27464285714])
        instance_prob_fun = lambda s, x: self.instance_prob[:x.shape[0]]
        with mock.patch('featmultinomial.FeatMultinomalNB.predict_proba',
                        return_value=self.instance_class_prob) as mock_pred:
            with mock.patch('featmultinomial.FeatMultinomalNB.instance_proba',
                            new=instance_prob_fun) as mock_inst_p:
                self.pipe._expectation_maximization()
                np.testing.assert_array_almost_equal(
                    self.pipe.classifier.class_log_prior_,
                    np.log(expected)
                )

    def test_em_sum_to_one(self):
        """Checks that both parameters estimated by the em step sums one."""
        self.pipe._expectation_maximization()
        self.assertAlmostEqual(
            (np.exp(self.pipe.classifier.class_log_prior_)).sum(), 1
        )
        np.testing.assert_array_almost_equal(
            (np.exp(self.pipe.classifier.feature_log_prob_)).sum(axis=1),
            np.ones(2)
        )

    def test_get_corpus(self):
        """Test the three corpus loaded from files."""
        self.assertEqual(len(self.pipe.training_corpus), len(X))
        self.assertEqual(len(self.pipe.test_corpus), len(T_vectors))
        self.assertEqual(len(self.pipe.unlabeled_corpus), len(U_vectors))
        self.assertEqual(len(self.pipe.user_corpus), 0)

    def test_set_config(self):
        """Each configuration must be set as attribute if not None."""
        for key, value in testing_config.items():
            if value is not None:
                self.assertTrue(hasattr(self.pipe, key))
                self.assertEqual(getattr(self.pipe, key), value)
            else:
                self.assertFalse(hasattr(self.pipe, key))

    def test_get_next_features(self):
        """Tests if the features are selected in order of IG for the class."""
        self.pipe.classifier.feat_information_gain = np.array([2, 0, 1])
        feat_indexes = self.pipe.get_next_features(class_number=0)
        self.assertEqual(feat_indexes, [2, 1])

    def test_get_next_features_repeated(self):
        """If the features where labeled, don't ask for them again."""
        self.pipe.classifier.feat_information_gain = np.array([2, 0, 1])
        self.pipe.asked_features[0][0] = True
        self.pipe.asked_features[0][2] = True
        feat_indexes = self.pipe.get_next_features(class_number=0)
        self.assertEqual(feat_indexes, [1])

    def test_handle_feature_prediction(self):
        """Positive and negative examples must be added to aked_features.

        For positive examples, the user_features must be changed.
        """
        class_number = 0
        self.pipe.handle_feature_prediction(class_number, full_set=[0, 1, 2],
                                            prediction=[1])
        self.assertEqual(self.pipe.user_features[0][1],
                         self.pipe.classifier.alpha + self.pipe.feature_boost,
                         msg='Bad Positive Example')
        self.assertEqual(self.pipe.user_features[0][0],
                         self.pipe.classifier.alpha,
                         msg='Change in non labeled feature')
        self.assertTrue(np.all(self.pipe.user_features[1] ==
                               self.pipe.classifier.alpha),
                        msg='Change in non labeled feature')
        self.assertTrue(self.pipe.asked_features[class_number][0])
        self.assertTrue(self.pipe.asked_features[class_number][1])
        self.assertTrue(self.pipe.asked_features[class_number][2])

    def test_get_next_instance(self):
        """Checks next instance selection using entropy.
        -E(1) = 0.5*log(0.5) + 0.5*log(0.5) = -0.6931471805599453
        -E(2) = 0.25*log(0.25) + 0.75*log(0.75) = -0.5623351446188083
        -E(3) = 0.7*log(0.7) + 0.3*log(0.3) = -0.6108643020548935
        -E(4) = 0.1*log(0.1) + 0.9*log(0.9) = -0.3250829733914482
        -E(5) = 0.8*log(0.8) + 0.2*log(0.2) = -0.5004024235381879
        """
        with mock.patch('featmultinomial.FeatMultinomalNB.predict_proba',
                        return_value=self.instance_class_prob) as mock_method:
            indexes = []
            while len(self.pipe.unlabeled_corpus) != 0:
                indexes.append(self.pipe.get_next_instance())
                self.pipe.unlabeled_corpus.pop_instance(indexes[-1])
            rigth_order = [3, 3, 1, 1, 0]
            self.assertEqual(indexes, rigth_order)

            self.assertIsNone(self.pipe.get_next_instance())


if __name__ == '__main__':
    unittest.main()
