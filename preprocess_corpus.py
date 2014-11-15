"""
Usage:
    preprocess_corpus.py <module_name> [<args> ...]
"""

# If module_name is all, then it will run all the preprocess modules in order
# on a corpus of instances. The input file must have instances in natural
# language separated in different lines.

from docopt import docopt
import preprocess

if __name__ == '__main__':
    opts = docopt(__doc__, version=0.1)
    module_name = opts['<module_name>']
    args = opts['<args>']
    # if module_name == 'all':
    #     if not len(args):
    #         return
    #     run_pipe(args[0])
    module = getattr(preprocess, module_name)
    module.process_corpus(*args)