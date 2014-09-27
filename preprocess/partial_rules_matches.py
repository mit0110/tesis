"""
Applies the quepy tagger to the questions of the corpus.

Usage:
    partial_rules_matches.py <input_filename> <output_filename>

The input file must have a pickled list of dictionaries with keys 'questions'.
Each question must be a list of quepy.Word that can be match with a quepy regex.

The output file will have the same structure replacing each questions by a tuple
where the first element is the original question and the second one a list of
the partially matched rules.
"""
import pickle
import refo

from docopt import docopt
from quepy import install

freebase_app = install('quepyapp_freebase')
_EOL = None


def process_corpus(input_filename, output_filename):
    input_f = open(input_filename, 'r')
    original_corpus = pickle.load(input_f)
    input_f.close()

    for instance in original_corpus:
        words = instance['question']
        rules = []
        for regex in freebase_app.partial_rules:
            match = refo.match(regex + refo.Literal(_EOL), words + [_EOL])
            if match:
                rules.append(repr(regex))
        instance['question'] = (instance['question'], rules)

    output_f = open(output_filename, 'w')
    pickle.dump(original_corpus, output_f)
    output_f.close()


if __name__ == '__main__':
    opts = docopt(__doc__, version=0.1)
    input_filename = opts['<input_filename>']
    output_filename = opts['<output_filename>']
    process_corpus(input_filename, output_filename)
