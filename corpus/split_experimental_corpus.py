from corpus import Corpus
from collections import defaultdict
import pickle

original_quepy_corpus = 'original_quepy_corpus.pickle'
original_unlabeled_corpus = 'original_unlabeled_corpus.pickle'


original_quepy_corpus_f = open(original_quepy_corpus, 'r')
original_unlabeled_corpus_f = open(original_unlabeled_corpus, 'r')

q_corpus = pickle.load(original_quepy_corpus_f)
original_quepy_corpus_f.close()
unrecognized_q = [q for q in q_corpus if not q['recognized']]
recognized_q = [q for q in q_corpus if q['recognized']]

o_u_corpus = pickle.load(original_unlabeled_corpus_f)
original_unlabeled_corpus_f.close()
unrecognized_q += [q for q in o_u_corpus if 'target' in q]
unlabeled_q = [q for q in o_u_corpus if 'target' not in q]

def separate_by_class(corpus):
    q_by_class = defaultdict(lambda: [])
    for q in corpus:
        if q['target'][0] != '':
            q_by_class[q['target'][0]].append(q)
    return q_by_class

u_by_class = separate_by_class(unrecognized_q)
r_by_class = separate_by_class(recognized_q)
tr_corpus = []
u_corpus = []
te_corpus = []

print [(k,len(r_by_class[k])) for k in r_by_class]

accepted_classes = [k for k in r_by_class
                    if len(r_by_class[k]) + len(u_by_class[k]) >= 6]
print accepted_classes
for k in r_by_class.keys():
    if not k in accepted_classes:
        r_by_class.pop(k)
        u_by_class.pop(k)

# Testing corpus
for q, v in r_by_class.items():
    original_len = len(v)
    for i in range(max(original_len/2, 1)):
        if v:
            te_corpus.append(v.pop())

print "Testing corpus: {} recognized instances".format(len(te_corpus))
for q, v in u_by_class.items():
    original_len = len(v)
    for i in range(original_len/4):
        if v:
            te_corpus.append(v.pop())

print "Testing corpus: {} total instances".format(len(te_corpus))
sep_te_corpus = separate_by_class(te_corpus)
print "Numbers of classes in Testing corpus ", len(sep_te_corpus)

# Training corpus
tr_classes = []
for q, v in r_by_class.items():
    if v:
        tr_corpus.append(v.pop())
        tr_classes.append(q)
print "Training corpus {} recognized questions".format(len(tr_corpus))

for q, v in u_by_class.items():
    if v and q not in tr_classes:
        tr_corpus.append(v.pop())
print "Training corpus total instances ", len(tr_corpus)
sep_tr_corpus = separate_by_class(tr_corpus)

# Unlabeled corpus
u_corpus += reduce(lambda x, y: x + y, r_by_class.values(), [])
print "Unlabeled corpus {} recognized instances".format(len(u_corpus))
# Add only a few of the "other" class
max_other = sorted([len(u_by_class[k]) for k in u_by_class])[-2]
for k, val in u_by_class.items():
    if k == 'other':
        u_corpus += val[:max_other]
    elif k != 'other':
        u_corpus += val
# u_corpus = reduce(lambda x, y: x + y, u_by_class.values(), [])


print "Unlabeled corpus total instances ", len(u_corpus)
sep_u_corpus = separate_by_class(u_corpus)

for k in sep_te_corpus.keys():
    r = [k, len(sep_te_corpus[k]), len(sep_tr_corpus[k]), len(sep_u_corpus[k])]
    print '{0} & {1} & {2} & {3} \\\\'.format(*r)

# Output files
test_corpus = 'experimental2/test_corpus.pickle'
training_corpus = 'experimental2/training_corpus.pickle'
unlabeled_corpus = 'experimental2/unlabeled_corpus.pickle'

test_corpus_f = open(test_corpus, 'w')
training_corpus_f = open(training_corpus, 'w')
unlabeled_corpus_f = open(unlabeled_corpus, 'w')

pickle.dump(te_corpus, test_corpus_f)
pickle.dump(tr_corpus, training_corpus_f)
pickle.dump(u_corpus, unlabeled_corpus_f)

test_corpus_f.close()
training_corpus_f.close()
unlabeled_corpus_f.close()