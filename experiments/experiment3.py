
"""Experiment number one

Run the boostrap for features and instances given the user the choice for one
of them after each labeling.
"""

import numpy as np
from termcolor import colored
from base_experiment import BaseExperiment
from metrics import (LearningCurve, PrecisionRecall, KappaStatistic,
                     PrecisionRecallCurve, ConfusionMatrix)
from random import randint, sample


def get_next_instance_max_entropy(self):
    """Selects a random instance from the unlabeled_corpus.

    Returns:
        An interger, the index of the instance.
    """
    if len(self.unlabeled_corpus) == 0:
        return None
    if self._retrained:
        self.u_clasifications = self.classifier.predict_proba(
            self.unlabeled_corpus.instances
        )
        entropy = self.u_clasifications * np.log(self.u_clasifications)
        entropy = entropy.sum(axis=1)
        entropy *= -1
        self.unlabeled_corpus.add_extra_info('entropy', entropy.tolist())

        self._retrained = False
    # Select the instance
    max_entropy = max(self.unlabeled_corpus.extra_info['entropy'])
    return self.unlabeled_corpus.extra_info['entropy'].index(max_entropy)


class Experiment3(BaseExperiment):
    def __init__(self, ActivePipeline):
        super(Experiment3, self).__init__(ActivePipeline)
        self.number = 3
        self.description = ("Active Learning on instaces using max entropy.")
        self.cycle_len = 1
        # Active learning instance selection function
        self.pipe_class.get_next_instance = get_next_instance_max_entropy

    def run(self):
        print "Running experiment number {0}: {1}".format(self.number,
                                                          self.description)
        self.pipe = self.pipe_class(emulate=True, **self.experiment_config)
        num_answers = 0
        while num_answers < self.max_answers:
            num_answers += self.pipe.instance_bootstrap(
                self.get_labeled_instance, max_iterations=self.cycle_len
            )
            print "{} of {} answers".format(num_answers, self.max_answers)
            self.pipe._train()
            self.pipe._expectation_maximization()
        self.get_name()
        self.save_session()
