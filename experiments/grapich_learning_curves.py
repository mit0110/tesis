"""
Usage:
    graphic_learning_curve.py (<file> <name>) ...

Creates a grapich with the learning curves on the files passed as parameters
"""

from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
from matplotlib import rc

# font = {'size': 15}

# rc('font', **font)


styles = ['r-', 'k--', 'b-.', 'm:', 'g_']
files = ['experiments/results/experiment14/learningcurve-inst105-featb100-350',
         'experiments/results/experiment15/learningcurve-inst105-featb100-350',
         'experiments/results/experiment16/learningcurve-inst105-featb100-350-avg'
         ]
names = ['Mayor IG', 'Menor IG', 'Aleatorio']

lines = []
for index, filename in enumerate(files):
    f = open(filename, 'r')
    dots = f.read().split('\n')
    dots = [d.split() for d in dots if d]
    x_dots = np.array([int(d[0])-104 for d in dots])  # HACKY
    y_dots = np.array([float(d[1]) for d in dots])
    f2 = interp1d(x_dots, y_dots, kind='cubic')
    xnew = np.linspace(min(x_dots), max(x_dots), 20)
    # lines += plt.plot(xnew, f2(xnew))
    lines += plt.plot(xnew, f2(xnew), styles[index])
    f.close()
for line in lines:
    plt.setp(line, linewidth=2)
plt.legend(lines, names, loc=4)
plt.axis([0, 360, 0, 1])
plt.xlabel('Numero de caracteristicas')
plt.ylabel('Accuracy')
plt.title('Accuracy en funcion del numero de caracteristicas etiquetadas')
plt.savefig('test1')
