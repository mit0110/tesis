"""
Usage:
    run_metric.py <metric_name> <filename>

There must be a corresponding class named metric_name. filename must contain
a pickled session.
"""

from docopt import docopt
from experiments import metrics
import pickle

if __name__ == '__main__':
    opts = docopt(__doc__, version=0.1)
    metric_name = opts['<metric_name>']
    filename = opts['<filename>']
    metric = getattr(metrics, metric_name)()
    f = open(filename, 'r')
    metric.session = pickle.load(f)
    f.close()
    metric.get_from_session()
    print metric.info