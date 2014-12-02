"""
Usage:
    graphic_learning_curve.py (<file> <name>) ...

Creates a grapich with the learning curves on the files passed as parameters
"""

from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import numpy as np


styles = ['r-o', 'b-s', 'g-^', '--']
files = ['experiments/results/experiment8/recognitioncurve-long-step1-boost1',
         'experiments/results/experiment8/recognitioncurve-long-step1-boost10',
         'experiments/results/experiment8/recognitioncurve-long-step1-boost30',
         'experiments/results/experiment8/recognitioncurve-long-step1-boost50',
         'experiments/results/experiment8/recognitioncurve-long-step1-boost80',
         'experiments/results/experiment8/recognitioncurve-long-step1-boost100',
         'experiments/results/experiment8/recognitioncurve-long-step1-boost150',
         'experiments/results/experiment8/recognitioncurve-long-step1-boost200']
names = ['1', '10', '30', '50', '80', '100', '150', '200']

lines = []
for index, filename in enumerate(files):
    f = open(filename, 'r')
    dots = f.read().split('\n')
    dots = [d.split() for d in dots if d]
    x_dots = np.array([int(d[0])-102 for d in dots])  # HACKY
    y_dots = np.array([float(d[1]) for d in dots])
    f2 = interp1d(x_dots, y_dots, kind='cubic')
    xnew = np.linspace(min(x_dots), max(x_dots), 90)
    lines += plt.plot(xnew, f2(xnew), styles[index])
plt.legend(lines, names)
plt.axis([0, 360, 0, 1])
plt.xlabel('Cantidad de instancias')
plt.ylabel('Reconocimiento')
plt.title('Reconocimiento en funcion de la cantidad de instancias')
plt.savefig('test1')

