
"""Experiment number one

Run the boostrap for features and instances given the user the choice for one
of them after each labeling.
"""

from termcolor import colored
from base_experiment import BaseExperiment
from metrics import (LearningCurve, PrecisionRecall, KappaStatistic,
                     PrecisionRecallCurve, ConfusionMatrix)
from random import randint, sample


class Experiment2(BaseExperiment):
    def __init__(self, ActivePipeline):
        super(Experiment2, self).__init__(ActivePipeline)
        self.number = 2
        self.description = ("Active Learning on instaces using entropy.")
        self.max_answers = 497
        self.cycle_len = 10
        self.metrics = [LearningCurve(), PrecisionRecall(), KappaStatistic(),
                        PrecisionRecallCurve(), ConfusionMatrix()]
        # Active learning instance selection function
        # self.pipe_class.get_next_instance = get_next_instance_random
        # Active learning class selection function
        # self.pipe_class.get_class_options = lambda s: s.classes
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
