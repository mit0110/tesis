
"""Experiment number one

Run the boostrap for features and instances given the user the choice for one
of them after each labeling.
"""

from termcolor import colored
from base_experiment import BaseExperiment
from metrics import LearningCurve, PrecisionRecall
from random import randint, sample


def get_next_instance_random(self):
    """Selects a random instance from the unlabeled_corpus.

    Returns:
        An interger, the index of the instance.
    """
    if not len(self.unlabeled_corpus):
        return None
    index = randint(0, len(self.unlabeled_corpus))
    return index


def get_next_features_random(self, class_index):
    """Selects a random set of features.

    Returns:
        A list of intergers. Each number is the index of a feature"""
    return sample(range(self.n_feat), self.number_of_features)


class Experiment1(BaseExperiment):
    def __init__(self, ActivePipeline):
        super(Experiment1, self).__init__(ActivePipeline)
        self.number = 1
        self.description = ("No active learning. Selecting instances and "
                            "features randomly.")
        self.max_answers = 300
        self.cycle_len = 1
        self.metrics = [LearningCurve(), PrecisionRecall(), KappaStatistic(),
                        PrecisionRecallCurve()]
        # Active learning instance selection function
        self.pipe_class.get_next_instance = get_next_instance_random
        # Active learning class selection function
        self.pipe_class.get_class_options = lambda s: s.classes
        self.experiment_config = {
            'u_corpus_f': 'corpus/experimental/unlabeled_new_corpus.pickle',
            'test_corpus_f': 'corpus/experimental/test_new_corpus.pickle',
            'training_corpus_f': 'corpus/experimental/training_new_corpus.pickle',
            'feature_corpus_f': 'corpus/experimental/feature_corpus.pickle',
        }

    def run(self):
        print "Running experiment number {0}: {1}".format(self.number,
                                                          self.description)
        self.pipe = self.pipe_class(emulate=True, **self.experiment_config)
        num_answers = 0
        while num_answers < self.max_answers:
            print colored('\n'.join(['*' * 79] * 10), 'green')
            print colored('Instance labeling', 'red', 'on_white',
                          attrs=['bold'])
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