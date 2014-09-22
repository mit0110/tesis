"""
Applies the quepy tagger to the questions of the corpus.

Usage:
    preprocess.py <input_filename> <output_filename>

The input file must have a pickled list of dictionaries with keys 'questions'.

The output file will have the same structure replacing each questions by a list
of instances of quepy.tagger.Word
"""
import pickle

from docopt import docopt
from quepy.tagger import get_tagger

tagger = get_tagger()


def process_corpus(input_filename, output_filename):
    input_f = open(input_filename, 'r')
    original_corpus = pickle.load(input_f)
    input_f.close()

    for instance in original_corpus:
        question = instance['question']
        instance['question'] = tagger(unicode(question))

    output_f = open(output_filename, 'w')
    pickle.dump(original_corpus, output_f)
    output_f.close()


if __name__ == '__main__':
    opts = docopt(__doc__, version=0.1)
    input_filename = opts['<input_filename>']
    output_filename = opts['<output_filename>']
    process_corpus(input_filename, output_filename)