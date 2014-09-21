import collections
import pickle
import random

f_original = open('original_annotated_corpus.pickle', 'r')
original_questions = pickle.load(f_original)
f_original.close()


not_recognized_questions = filter(lambda x: x['recognized'] == 'n', original_questions)
recognized_questions = filter(lambda x: x['recognized'] == 's', original_questions)

random.shuffle(not_recognized_questions)
needed_questions = int(len(original_questions) * 0.1)

test_questions = not_recognized_questions[:needed_questions + 1]
training_questions = not_recognized_questions[needed_questions + 1:] + recognized_questions

f_training = open('annotated_corpus.pickle', 'w')
pickle.dump(training_questions, f_training)
f_training.close()

f_testing = open('test_corpus.pickle', 'w')
pickle.dump(test_questions, f_testing)
f_testing.close()