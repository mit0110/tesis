
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


class Experiment6(BaseExperiment):
    def __init__(self, ActivePipeline):
        super(Experiment6, self).__init__(ActivePipeline)
        self.number = 6
        self.description = ("Feature active learning.")
        self.max_feat_answers = 350
        self.cycle_len = 10

    def run(self):
        print "Running experiment number {0}: {1}".format(self.number,
                                                          self.description)
        self.pipe = self.pipe_class(emulate=True, **self.experiment_config)
        num_answers = 0
        self.pipe.instance_bootstrap(
            self.get_labeled_instance, max_iterations=self.max_answers
        )
        while num_answers < self.max_feat_answers:
            # print colored('\n'.join(['*' * 79] * 10), 'green')
            # print colored('Instance labeling', 'red', 'on_white',
            #               attrs=['bold'])
            print "{} of {} answers".format(num_answers, self.max_feat_answers)
            print colored('\n'.join(['*' * 79] * 3), 'green')
            # print colored('Feature labeling', 'red', 'on_white',
            #               attrs=['bold'])
            num_answers += self.pipe.feature_bootstrap(get_class,
                self.get_labeled_features, max_iterations=self.cycle_len
            )
            self.pipe._train()
            self.pipe._expectation_maximization()
        self.get_name()
        self.save_session()
        self.pipe.label_feature_corpus()
