"""Interface to query the named entities in the database.
"""
import pickle

NER_F = "data/celebrities.pickle"

class LiteralNER(object):
    def __init__(self):
        f = open(NER_F, 'r')
        self.data = pickle.load(f)
        f.close()

    def find_ne(self, sentence):
        """Returns a named entity and its types that is substring of sentence.

        Args:
            sentence: A string to match.
        """
        for word, types in self.data:
            if word in sentence:
                return str(word), [str(t) for t in types]
        return None