
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


class Experiment8(BaseExperiment):
    def __init__(self, ActivePipeline):
        super(Experiment8, self).__init__(ActivePipeline)
        self.number = 8
        self.description = ("Feature active learning over all corpus "
                            "with different boosts")
        self.unlabeled_corpus_len = 495
        self.max_answers = 300 + self.unlabeled_corpus_len
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
        while num_answers < self.max_answers:
            num_answers += self.pipe.instance_bootstrap(
                self.get_labeled_instance,
                max_iterations=self.unlabeled_corpus_len
            )
            print "{} of {} answers".format(num_answers, self.max_answers)
            print colored('\n'.join(['*' * 79] * 3), 'green')
            num_answers += self.pipe.feature_bootstrap(get_class,
                self.get_labeled_features, max_iterations=self.cycle_len
            )
            self.pipe._train()
            self.pipe._expectation_maximization()
        self.instance_name = 'long-step1-boost{}'.format(feature_boost)
        self.save_session()
        self.pipe.label_feature_corpus()