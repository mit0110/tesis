
"""Experiment number one

Run the boostrap for features and instances given the user the choice for one
of them after each labeling.
"""
import numpy as np
from termcolor import colored
from base_experiment import BaseExperiment
from featmultinomial import FeatMultinomalNB
from metrics import (LearningCurve, PrecisionRecall, KappaStatistic,
                     PrecisionRecallCurve)
from random import randint, sample
from sklearn.preprocessing import normalize

ACTUAL_CLASS = -1


def get_class(class_list):
    global ACTUAL_CLASS
    ACTUAL_CLASS = (ACTUAL_CLASS + 1) % len(class_list)
    return class_list[ACTUAL_CLASS]


class Experiment9(BaseExperiment):
    def __init__(self, ActivePipeline):
        super(Experiment9, self).__init__(ActivePipeline)
        self.number = 9
        self.description = ("Feature active learning over all corpus "
                            "using class prior")
        self.unlabeled_corpus_len = 107
        self.max_answers = 200 + self.unlabeled_corpus_len
        self.cycle_len = 1
        self.metrics = [LearningCurve(), PrecisionRecall(), KappaStatistic(),
                        PrecisionRecallCurve()]
        class_prior = np.zeros((29,))
        class_prior += 1
        class_prior[18] = 1  # Position of the class other
        class_prior = normalize(class_prior.reshape((1, 29)), norm='l1')[0]

        self.experiment_config = {
            'u_corpus_f': 'corpus/experimental/unlabeled_new_corpus_balanced.pickle',
            'test_corpus_f': 'corpus/experimental/test_new_corpus.pickle',
            'training_corpus_f': 'corpus/experimental/training_new_corpus.pickle',
            'feature_corpus_f': 'corpus/experimental/feature_corpus.pickle',
            'classifier': FeatMultinomalNB(class_prior=class_prior)
        }

    def run(self):
        print "Running experiment number {0}: {1}".format(self.number,
                                                          self.description)
        num_answers = 0
        self.pipe = self.pipe_class(emulate=True, **self.experiment_config)
        while num_answers < self.max_answers:
            num_answers += self.pipe.instance_bootstrap(
                self.get_labeled_instance,
                max_iterations=self.unlabeled_corpus_len
            )
            print "{} of {} answers".format(num_answers, self.max_answers)
            print colored('\n'.join(['*' * 79] * 3), 'green')
            num_answers += self.pipe.feature_bootstrap(get_class,
                lambda x, y: [], max_iterations=self.cycle_len
            )
            self.pipe._train()
            # self.pipe._expectation_maximization()
        self.get_name()
        self.save_session()