from corpus import Corpus
from collections import defaultdict
import numpy as np

co = Corpus()
co.load_from_file('experimental/unlabeled_new_corpus2.pickle')

def count_by_class(corpus):
    """Returns a dictionary with the number of instances by class"""
    result = defaultdict(lambda: 0)
    for target in co.primary_targets:
        result[target] += 1
    return result

c_by_class = count_by_class(co)

for k, v in c_by_class.items():
    print k, v

limit = sorted(c_by_class.values())[-2]
# limit = 10
to_remove = c_by_class['other'] - limit
to_remove = {k: c_by_class[k] - limit for k in c_by_class}
print to_remove, limit

for i in range(len(co)-1, 0, -1):
    target = co.primary_targets[i]
    if to_remove[target] > 0:
        co.pop_instance(i)
        to_remove[target] -= 1

c_by_class = count_by_class(co)

for k, v in c_by_class.items():
    print k, v

co.save_to_file('experimental/unlabeled_new_corpus_balanced2.pickle')