"""
Applies the quepy tagger to the questions of the corpus.

Usage:
    feature_extraction.py <l_input_fname> <u_input_fname> <output_filename>

Both input file must have a pickled list of dictionaries with keys 'questions'.
Each question will be procesed with the features in the file feature_extraction.

The output file will have a pickled dictiory with keys:
    instance_rpr: the natural language representation of each question.
    instance: an array with all the vectors.

"""
import pickle
from feature_extraction import get_features

from docopt import docopt


def process_corpus(l_input_filename, u_input_filename, output_filename):
    input_f = open(l_input_filename, 'r')
    original_corpus = pickle.load(input_f)
    input_f.close()

    input_f = open(u_input_filename, 'r')
    original_corpus += pickle.load(input_f)
    input_f.close()

    new_corpus = {'question_strings':[]}

    output_f = open(output_filename, 'w')
    pickle.dump(original_corpus, output_f)
    output_f.close()


if __name__ == '__main__':
    opts = docopt(__doc__, version=0.1)
    l_input_filename = opts['<l_input_fname>']
    u_input_filename = opts['<u_input_fname>']
    output_filename = opts['<output_filename>']
    process_corpus(l_input_filename, u_input_filename, output_filename)
