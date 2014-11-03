import unittest
from corpus import Corpus
from scipy.sparse import csr_matrix

def _eq_crs_matrix(m1, m2):
    return not (m1 != m2).todense().any()


class TestCorpus(unittest.TestCase):

    def setUp(self):
        self.co = Corpus()
        self.co.instances = csr_matrix([[1, 2, 3], [4, 5, 6]])
        self.co.full_targets = [[1], [2,3]]
        self.co.representations = ['representation1', 'representation2']
        self.co.calculate_primary_targets()

    def tearDown(self):
        self.assertTrue(self.co.check_consistency())

    def test_load_and_save(self):
        """Load and save functions must be inverses."""
        filename = 'testing_file'
        self.co.save_to_file(filename)
        new_co = Corpus()
        new_co.load_from_file(filename)
        self.assertTrue(_eq_crs_matrix(new_co.instances, self.co.instances))
        for index in range(len(self.co)):
            self.assertEqual(self.co.full_targets[index],
                             new_co.full_targets[index])
            self.assertEqual(self.co.representations[index],
                             new_co.representations[index])
        self.assertIsNotNone(new_co.primary_targets)

    def test_add_instance(self):
        self.co.add_instance([2, 3, 4], [2], 'representation3')
        self.assertEqual(len(self.co), 3)
        self.assertTrue(_eq_crs_matrix(csr_matrix([2, 3, 4]),
                                       self.co.instances[-1]))

    def test_pop_first_instance(self):
        result = self.co.pop_instance(0)
        # check result
        self.assertEqual(len(result), 3)
        self.assertTrue(_eq_crs_matrix(csr_matrix([1, 2, 3]), result[0]))
        self.assertEqual(result[1], [1])
        self.assertEqual(result[2], 'representation1')
        #check corpus
        self.assertEqual(len(self.co), 1)
        self.assertTrue(_eq_crs_matrix(csr_matrix([4, 5, 6]),
                                       self.co.instances[0]))
        self.assertEqual(self.co.full_targets[0], [2, 3])
        self.assertEqual(self.co.representations[0], 'representation2')

    def test_pop_last_instance(self):
        result = self.co.pop_instance(1)
        # check result
        self.assertEqual(len(result), 3)
        self.assertTrue(_eq_crs_matrix(csr_matrix([4, 5, 6]), result[0]))
        self.assertEqual(result[1], [2, 3])
        self.assertEqual(result[2], 'representation2')
        #check corpus
        self.assertEqual(len(self.co), 1)
        self.assertTrue(_eq_crs_matrix(csr_matrix([1, 2, 3]),
                                       self.co.instances[0]))
        self.assertEqual(self.co.full_targets[0], [1])
        self.assertEqual(self.co.representations[0], 'representation1')

    def test_pop_middle_instace(self):
        self.co.add_instance([2, 3, 4], [2], 'representation3')
        result = self.co.pop_instance(1)

        self.assertEqual(len(self.co), 2)
        self.assertEqual(len(result), 3)
        self.assertTrue(_eq_crs_matrix(csr_matrix([4, 5, 6]), result[0]))
        self.assertEqual(result[1], [2, 3])
        self.assertEqual(result[2], 'representation2')

    def test_pop_last_instance(self):
        self.co.pop_instance(1)
        self.co.pop_instance(0)

        self.assertEqual(len(self.co), 0)

    def test_concatenate_corpus(self):
        new_corpus = Corpus()
        new_corpus.add_instance([2, 3, 4], [2], 'representation3')
        new_corpus.add_instance([10, 4, 4], [1, 1, 2], 'representation3')
        new_corpus.calculate_primary_targets()

        self.co.concetenate_corpus(new_corpus)
        self.assertEqual(len(self.co), 4)

    def test_concatenate_empty_corpus(self):
        new_corpus = Corpus()
        self.co.concetenate_corpus(new_corpus)
        self.assertEqual(len(self.co), 2)

    def test_calculate_primary_targets(self):
        self.assertEqual(self.co.primary_targets, [1,2])

    def test_primary_targets_none(self):
        self.co.add_instance([0, 2, 3], [], 'r')
        self.co.calculate_primary_targets()
        self.assertEqual(self.co.primary_targets, [1, 2, None])

    def test_primary_targets_mode(self):
        self.co.add_instance([0, 2, 3], [1, 4, 4, 3, 3], 'r')
        self.co.calculate_primary_targets()
        self.assertEqual(self.co.primary_targets, [1, 2, 3])

    def test_len(self):
        self.assertEqual(len(self.co), 2)

    def test_len_empty(self):
        new_corpus = Corpus()
        self.assertEqual(len(new_corpus), 0)

if __name__ == '__main__':
    unittest.main()