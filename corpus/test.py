import unittest
from corpus import Corpus
from scipy.sparse import csr_matrix

class TestCorpus(unittest.TestCase):

    def setUp(self):
        self.co = Corpus()
        self.co.instances = csr_matrix([[1, 2, 3], [4, 5, 6]])
        self.co.targets = [[1], [2,3]]
        self.co.representations = ['representation1', 'representation2']

    def test_load_and_save(self):
        """Load and save functions must be inverses."""
        filename = 'testing_file'
        self.co.save_to_file(filename)
        new_co = Corpus()
        new_co.load_from_file(filename)
        eq = (new_co.instances != self.co.instances).todense()
        self.assertFalse(eq.any())
        for index in range(len(self.co)):
            self.assertEqual(self.co.targets[index], new_co.targets[index])
            self.assertEqual(self.co.representations[index],
                             new_co.representations[index])

    def test_add_instance(self):
        self.co.add_instance([2, 3, 4], [2], 'representation3')
        self.assertEqual(len(self.co), 3)


if __name__ == '__main__':
    unittest.main()