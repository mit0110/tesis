"""
SOURCE FILES
-- original_quepy_corpus.pickle
-- original_unlabeled_corpus.pickle
Both have a list of dictionaries with keys 'question' and 'target'


GENERATED FILES
test_corpus.pickle
-- 50 quepy questions
-- 200 other labeled questions

training_corpus.pickle
-- remaining quepy questions
-- 100 other labeled questions

unlabeled_corpus.pickle
-- All the other remaining questions

Running this script would change the values according to random shuffles!
"""
import pickle
import random

# Data sources
original_quepy_corpus = 'original_quepy_corpus.pickle'
original_unlabeled_corpus = 'original_unlabeled_corpus.pickle'

# Output files
test_corpus = 'test_corpus.pickle'
training_corpus = 'training_corpus.pickle'
unlabeled_corpus = 'unlabeled_corpus.pickle'


def main():
    original_quepy_corpus_f = open(original_quepy_corpus, 'r')
    original_unlabeled_corpus_f = open(original_unlabeled_corpus, 'r')

    test_corpus_f = open(test_corpus, 'w')
    training_corpus_f = open(training_corpus, 'w')
    unlabeled_corpus_f = open(unlabeled_corpus, 'w')

    test_results = []
    training_results = []
    unlabeled_results = []

    # Quepy
    q_corpus = pickle.load(original_quepy_corpus_f)
    original_quepy_corpus_f.close()
    unrecognized_q = [q for q in q_corpus if not q['recognized']]
    recognized_q = [q for q in q_corpus if q['recognized']]
    test_results = unrecognized_q[:50]
    training_results = recognized_q + unrecognized_q[50:]
    print "Quepy questions", len(q_corpus)
    print "\tRecognized", len(recognized_q)
    print "\tUnrecognized", len(unrecognized_q)

    quepy_classes = [q['target'] for q in q_corpus]

    # Unlabeled
    o_u_corpus = pickle.load(original_unlabeled_corpus_f)
    original_unlabeled_corpus_f.close()
    labeled_q = [q for q in o_u_corpus if 'target' in q]
    unlabeled_q = [q for q in o_u_corpus if 'target' not in q]
    print "Other questions", len(o_u_corpus)
    print "\tLabeled", len(labeled_q)
    print "\tUnlabeled", len(unlabeled_q)

    random.shuffle(labeled_q)
    test_results += labeled_q[:200]
    training_results += labeled_q[200:300]
    unlabeled_results = unlabeled_q + labeled_q[300:]

    print ''
    print "Test corpus", len(test_results)
    print "Training corpus", len(training_results)
    print "Unlabeled corpus", len(unlabeled_results)
    assert(len(test_results) + len(training_results) + len(unlabeled_results)
           == len(o_u_corpus) + len(q_corpus))

    pickle.dump(test_results, test_corpus_f)
    test_corpus_f.close()
    pickle.dump(training_results, training_corpus_f)
    training_corpus_f.close()
    pickle.dump(unlabeled_results, unlabeled_corpus_f)
    unlabeled_corpus_f.close()

    print ''
    print "New classes found"
    wrong_unlabeled_classes = [q['target'] for q in labeled_q
                               if q['target'] not in quepy_classes and
                               q['target'] != 'other']
    print set(wrong_unlabeled_classes)

if __name__ == '__main__':
    main()