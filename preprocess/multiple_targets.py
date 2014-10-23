"""
Searchs for possible named entities in the question.

Usage:
    multiple_targets.py <input_filename> <output_filename>

The input file must have a pickled list of dictionaries with keys 'target'.

The output file will have the same structure but instead of the value of
'target' will be a list with this value.
"""
import pickle

from docopt import docopt

def process_corpus(input_filename, output_filename):
    input_f = open(input_filename, 'r')
    original_corpus = pickle.load(input_f)
    input_f.close()
    count = 0
    for instance in original_corpus:
        if 'target' in instance:
            instance['target'] = [instance['target']]
    output_f = open(output_filename, 'w')
    pickle.dump(original_corpus, output_f)
    output_f.close()


if __name__ == '__main__':
    opts = docopt(__doc__, version=0.1)
    input_filename = opts['<input_filename>']
    output_filename = opts['<output_filename>']
    process_corpus(input_filename, output_filename)
