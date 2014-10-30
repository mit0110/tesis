"""
Applies the quepy tagger to the questions of the corpus.

Both input file must have a pickled list of dictionaries with keys 'questions'.
Each question will be procesed with the features in the file feature_extraction.

The output file will have a pickled instance of Corpus

"""

import pickle
from feature_extraction import get_features
from corpus import Corpus

from docopt import docopt


def process_corpus(l_input_filename, u_input_filename, output_filename):
    input_f = open(l_input_filename, 'r')
    l_original_corpus = pickle.load(input_f)
    input_f.close()

    input_f = open(u_input_filename, 'r')
    u_original_corpus = pickle.load(input_f)
    input_f.close()

    vect = get_features()
    l_questions = [d['question'] for d in l_original_corpus]
    u_questions = [d['question'] for d in u_original_corpus]

    vect.fit(l_questions + u_questions)
    corpus = Corpus()
    corpus.instances = vect.transform(l_questions + u_questions)
