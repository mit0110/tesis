"""
Usage:
    preprocess_corpus.py <module_name> [<args> ...]

"""
from docopt import docopt
import preprocess

if __name__ == '__main__':
    opts = docopt(__doc__, version=0.1)
    module_name = opts['<module_name>']
    args = opts['<args>']
    module = getattr(preprocess, module_name)
    module.process_corpus(*args)