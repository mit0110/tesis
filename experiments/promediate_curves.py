"""
Usage:
    graphic_learning_curve.py <file> ...

Creates a grapich with the learning curves on the files passed as parameters
"""

from docopt import docopt
import numpy as np


if __name__ == '__main__':
    opts = docopt(__doc__, version=0.1)
    files = opts['<file>']

    result = []
    for filename in files:
        f = open(filename, 'r')
        dots = f.read().split('\n')
        f.close()
        y_coord = [int(d.split('\t')[0]) for d in dots if d]
        dots = [float(d.split('\t')[1]) for d in dots if d]
        result.append(dots)
    result = np.array(result)
    result = result.mean(axis=0)

    for index, mean in enumerate(result.T):
        print y_coord[index], mean

