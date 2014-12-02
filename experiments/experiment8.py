
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
        self.max_feat_answers = 350
        self.cycle_len = 1

    def run(self):
        print "Running experiment number {0}: {1}".format(self.number,
                                                          self.description)
        feature_boost = 120
        num_answers = 0
        self.experiment_config['feature_boost'] = feature_boost
        self.pipe = self.pipe_class(emulate=True, **self.experiment_config)
        self.pipe.instance_bootstrap(
            self.get_labeled_instance, max_iterations=self.max_answers
        )
        while num_answers < self.max_feat_answers:
            print "{} of {} answers".format(num_answers, self.max_feat_answers)
            print colored('\n'.join(['*' * 79] * 3), 'green')
            num_answers += self.pipe.feature_bootstrap(get_class,
                lambda x, y: [], max_iterations=self.cycle_len
            )
            self.pipe._train()
            self.pipe._expectation_maximization()
        self.instance_name = 'long-step1-boost{}'.format(feature_boost)
        self.save_session()
        # self.pipe.label_feature_corpus()