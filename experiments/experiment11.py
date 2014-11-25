
"""Experiment number one

Run the boostrap for features and instances given the user the choice for one
of them after each labeling.
"""

from termcolor import colored
from base_experiment import BaseExperiment
from metrics import (LearningCurve, PrecisionRecall, KappaStatistic,
                     PrecisionRecallCurve)
from random import randint, sample

ACTUAL_CLASS = -1


def get_class(class_list):
    global ACTUAL_CLASS
    ACTUAL_CLASS = (ACTUAL_CLASS + 1) % len(class_list)
    return class_list[ACTUAL_CLASS]


def get_next_features(self, class_number):
    """Selects features with low IG"""
    selected_f_pos = self.classifier.feature_count_[class_number].argsort()
    def non_seen_filter(i):
        return not self.asked_features[class_number][i]
    selected_f_pos = filter(non_seen_filter, selected_f_pos.tolist())
    selected_f_pos = selected_f_pos[:-(self.number_of_features+1):-1]
    def key_fun(i): return self.classifier.feat_information_gain[i]
    selected_f_pos.sort(key=key_fun)
    return selected_f_pos


class Experiment11(BaseExperiment):
    def __init__(self, ActivePipeline):
        super(Experiment11, self).__init__(ActivePipeline)
        self.number = 11
        self.description = ("Feature active learning over all corpus "
                            "first high IG and then low.")
        self.unlabeled_corpus_len = 495
        self.max_answers = 350 + self.unlabeled_corpus_len
        self.cycle_len = 1
        self.metrics = [LearningCurve(), PrecisionRecall(), KappaStatistic(),
                        PrecisionRecallCurve()]
        self.experiment_config = {
            'u_corpus_f': 'corpus/experimental/unlabeled_new_corpus.pickle',
            'test_corpus_f': 'corpus/experimental/test_new_corpus.pickle',
            'training_corpus_f': 'corpus/experimental/training_new_corpus.pickle',
            'feature_corpus_f': 'corpus/experimental/feature_corpus.pickle',
        }

    def run(self):
        print "Running experiment number {0}: {1}".format(self.number,
                                                          self.description)
        feature_boost = 50
        num_answers = 0
        self.experiment_config['feature_boost'] = feature_boost
        self.pipe = self.pipe_class(emulate=True, **self.experiment_config)
        change_point = self.unlabeled_corpus_len + 300
        num_answers += self.pipe.instance_bootstrap(
            self.get_labeled_instance,
            max_iterations=self.unlabeled_corpus_len
        )
        while num_answers < self.max_answers:
            print "{} of {} answers".format(num_answers, self.max_answers)
            print colored('\n'.join(['*' * 79] * 3), 'green')
            num_answers += self.pipe.feature_bootstrap(get_class,
                self.get_labeled_features, max_iterations=self.cycle_len
            )
            self.pipe._train()
            self.pipe._expectation_maximization()
            if num_answers > change_point:
                print "changing strategy"
                self.pipe.get_next_features = get_next_features
        self.instance_name = 'long-step1-boost{}'.format(feature_boost)
        self.save_session()
        self.pipe.label_feature_corpus()