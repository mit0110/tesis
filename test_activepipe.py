
import unittest
import numpy as np
import mock

from activepipe import ActivePipeline
from featureforge.vectorizer import Vectorizer
from corpus import Corpus
from scipy.sparse import csr_matrix

testing_config = {
    'features': Vectorizer([lambda x : x]),
    'em_adding_instances': 3,
    'u_corpus_f': 'test_files/unlabeled_corpus.pickle',
    'test_corpus_f': 'test_files/test_corpus.pickle',
    'training_corpus_f': 'test_files/training_corpus.pickle',
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

def _eq_crs_matrix(m1, m2):
    return not (m1 != m2).todense().any()


class TestActivePipe(unittest.TestCase):

    def setUp(self):
        # def mock_get_corpus_fun(self):
            # self.training_corpus = Corpus()
            # self.training_corpus.instances = csr_matrix(X)
            # self.training_corpus.full_targets = Y
            # self.training_corpus.representations = ['r'] * len(X)
            # self.training_corpus.calculate_primary_targets()
            # self.training_corpus.save_to_file('test_files/training_corpus.pickle')
            # self.unlabeled_corpus = Corpus()
            # self.unlabeled_corpus.instances = csr_matrix(U_vectors)
            # self.unlabeled_corpus.full_targets = [[]] * len(U_vectors)
            # self.unlabeled_corpus.representations = ['u'] * len(U_vectors)
            # self.unlabeled_corpus.calculate_primary_targets()
            # self.unlabeled_corpus.save_to_file('test_files/unlabeled_corpus.pickle')
            # self.user_corpus = Corpus()
            # self.test_corpus = Corpus()
            # self.test_corpus.instances = csr_matrix(T_vectors)
            # self.test_corpus.full_targets = [[1]] * len(T_vectors)
            # self.test_corpus.representations = ['u'] * len(T_vectors)
            # self.test_corpus.calculate_primary_targets()
            # self.test_corpus.save_to_file('test_files/test_corpus.pickle')

        # self.mock_get_corpus = mock.patch(
        #     'activepipe.ActivePipeline._get_corpus',
        #     mock_get_corpus_fun
        # )
        # self.mock_get_corpus.start()
        self.pipe = ActivePipeline(**testing_config)

    # def tearDown(self):
    #     self.mock_get_corpus.stop()

    def test_em_select(self):
        """Tests if the rigth instances are selected to add in training vectors.
        """
        self.pipe._expectation_maximization()
        self.assertTrue(_eq_crs_matrix(csr_matrix([1.0, 1.0, 1.0]),
                                       self.pipe.training_corpus.instances[-1]))
        self.assertTrue(_eq_crs_matrix(csr_matrix([1.0, 0.0, 0.0]),
                                       self.pipe.training_corpus.instances[-2]))
        self.assertTrue(_eq_crs_matrix(csr_matrix([2.0, 2.0, 2.0]),
                                       self.pipe.training_corpus.instances[-3]))

        self.assertEqual(len(self.pipe.unlabeled_corpus), 2)

    def test_get_corpus(self):
        """Test the three corpus loaded from files."""
        self.assertEqual(len(self.pipe.training_corpus), len(X))
        self.assertEqual(len(self.pipe.test_corpus), len(T_vectors))
        self.assertEqual(len(self.pipe.unlabeled_corpus), len(U_vectors))
        self.assertEqual(len(self.pipe.user_corpus), 0)


if __name__ == '__main__':
    unittest.main()
