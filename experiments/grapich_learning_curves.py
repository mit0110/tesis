"""
Usage:
    graphic_learning_curve.py (<file> <name>) ...

Creates a grapich with the learning curves on the files passed as parameters
"""

from docopt import docopt
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np


styles = ['r-o', 'b-s', 'g-^', '--']


if __name__ == '__main__':
    opts = docopt(__doc__, version=0.1)
    files = opts['<file>']
    names = opts['<name>']

    lines = []
    for index, filename in enumerate(files):
        f = open(filename, 'r')
        dots = f.read().split('\n')
        dots = [d.split('\t') for d in dots if d]
        x_dots = np.array([int(d[0]) for d in dots])
        y_dots = np.array([float(d[1]) for d in dots])
        f2 = interp1d(x_dots, y_dots, kind='cubic')
        xnew = np.linspace(min(x_dots), max(x_dots), 20)
        lines += plt.plot(xnew, f2(xnew), styles[index])
    plt.legend(lines, names)
    plt.xlabel('Cantidad de instancias')
    plt.ylabel('Reconocimiento')
    plt.title('Comparacion entre las estrategias de seleccion de instancias')
    plt.savefig('test1')

