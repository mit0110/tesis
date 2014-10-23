"""Interface to query the named entities in the database.
"""
import pickle

NER_F = "data/all_data.pickle"

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
        possible_matches = []
        for word, types in self.data.items():
            if word in sentence:
                possible_matches.append(word)
        possible_matches.sort(key = lambda s: len(s))
        if possible_matches:
            word = possible_matches[-1]
            return word, self.data[word]
        return None, None