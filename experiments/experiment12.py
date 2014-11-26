
"""Experiment number one

Run the boostrap for features and instances given the user the choice for one
of them after each labeling.
"""

from termcolor import colored
from base_experiment import BaseExperiment
from metrics import (LearningCurve, PrecisionRecall, KappaStatistic,
                     PrecisionRecallCurve, ConfusionMatrix)
from random import randint, sample
import numpy as np

def get_next_instance(self):
    """Selects the index of an unlabeled instance to be sent to the user.

    Uses min entropy.

    Returns:
        The index of an instance selected from the unlabeled_corpus.
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


class Experiment12(BaseExperiment):
    def __init__(self, ActivePipeline):
        super(Experiment12, self).__init__(ActivePipeline)
        self.number = 12
        self.description = ("Active Learning on instaces using min entropy.")
        self.max_answers = 107
        self.cycle_len = 1
        self.metrics = [LearningCurve(), PrecisionRecall(), KappaStatistic(),
                        PrecisionRecallCurve(), ConfusionMatrix()]
        # Active learning instance selection function
        # self.pipe_class.get_next_instance = get_next_instance_random
        # Active learning class selection function
        # self.pipe_class.get_class_options = lambda s: s.classes
        self.experiment_config = {
            'u_corpus_f': 'corpus/experimental/unlabeled_new_corpus_balanced.pickle',
            'test_corpus_f': 'corpus/experimental/test_new_corpus.pickle',
            'training_corpus_f': 'corpus/experimental/training_new_corpus.pickle',
            'feature_corpus_f': 'corpus/experimental/feature_corpus.pickle',
        }
        self.pipe_class.get_next_instance = get_next_instance

    def run(self):
        print "Running experiment number {0}: {1}".format(self.number,
                                                          self.description)
        self.pipe = self.pipe_class(emulate=True, **self.experiment_config)
        num_answers = 0
        while num_answers < self.max_answers:
            # print colored('\n'.join(['*' * 79] * 10), 'green')
            # print colored('Instance labeling', 'red', 'on_white',
            #               attrs=['bold'])
            num_answers += self.pipe.instance_bootstrap(
                self.get_labeled_instance, max_iterations=self.cycle_len
            )
            print "{} of {} answers".format(num_answers, self.max_answers)
            # print colored('\n'.join(['*' * 79] * 10), 'green')
            # print colored('Feature labeling', 'red', 'on_white',
            #               attrs=['bold'])
            # num_answers += self.pipe.feature_bootstrap(self.get_class,
            #     self.get_labeled_features, max_iterations=self.cycle_len
            # )
            self.pipe._train()
            self.pipe._expectation_maximization()
        self.get_name()
        self.save_session()
