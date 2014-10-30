"""
Searchs for possible named entities in the question.

Usage:
    NER.py <input_filename> <output_filename>

The input file must have a pickled list of dictionaries with keys 'questions'.
Each question must be a tuple where the first element is a list of quepy Words.

The output file will have the same structure adding two more fields into the
tuple, one for the name of the entity and other for a list of its types.
"""
import pickle

from docopt import docopt
from literal_ner import LiteralNER


def process_corpus(input_filename, output_filename):
    lner = LiteralNER()
    input_f = open(input_filename, 'r')
    original_corpus = pickle.load(input_f)
    input_f.close()
    count = 0
    for instance in original_corpus:
        words = ' '.join([w.token for w in instance['question'][0]])
        ne, types = lner.find_ne(words)
        if ne:
            count += 1
            try:
                types = [str(t) for t in types]
            except UnicodeEncodeError:
                print "error", types
                types = [str(t) for t in types[:-1]]
            instance['question'] = instance['question'][:-2] + (str(ne), tuple(types))
        else:
            instance['question'] = instance['question'][:-2] + ('None', ('None',))
    print "... {0} entities found in {1} questions".format(
        count, len(original_corpus)
    )
    output_f = open(output_filename, 'w')
    pickle.dump(original_corpus, output_f)
    output_f.close()


if __name__ == '__main__':
    opts = docopt(__doc__, version=0.1)
    input_filename = opts['<input_filename>']
    output_filename = opts['<output_filename>']
    process_corpus(input_filename, output_filename)

