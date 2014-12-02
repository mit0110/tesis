from corpus import Corpus
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report

co = Corpus()
c2 = Corpus()
ct = Corpus()

co.load_from_file('corpus/experimental2/unlabeled_new_corpus.pickle')
ct.load_from_file('corpus/experimental2/test_new_corpus.pickle')
c2.load_from_file('corpus/experimental2/training_new_corpus.pickle')

co.concetenate_corpus(c2)

mnb = MultinomialNB()
mnb.fit(co.instances, co.primary_targets)
print mnb.score(ct.instances, ct.primary_targets)
predicted_targets = mnb.predict(ct.instances)
print classification_report(ct.primary_targets, predicted_targets)
cm = confusion_matrix(ct.primary_targets, predicted_targets)
for index, row in enumerate(cm):
    print mnb.classes_[index], [(mnb.classes_[j], row[j])
                                for j in range(len(row))
                                if row[j]]

new_q = 0
for index, row in enumerate(cm):
    if mnb.classes_[index] != 'other':
        new_q += cm[index][index]
print new_q / float(cm.sum()-135)
