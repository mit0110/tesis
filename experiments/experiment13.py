
"""Experiment number one

Run the boostrap for features and instances given the user the choice for one
of them after each labeling.
"""

from termcolor import colored
from base_experiment import BaseExperiment
from metrics import (RecognitionVsFeatBoost, AccuracyVsFeatBoost,
                     KappaVsFeatBoost)
import numpy as np


class Experiment13(BaseExperiment):
    def __init__(self, ActivePipeline):
        super(Experiment13, self).__init__(ActivePipeline)
        self.number = 13
        self.description = ("Feature training with different number "
                            "of instances.")
        self.feat_boost = 100
        self.instance_number = 92
        self.cycle_len = 1

    def run(self):
        print "Running experiment number {0}: {1}".format(self.number,
                                                          self.description)
        self.pipe = self.pipe_class(emulate=True, **self.experiment_config)
        # Adding the features from the corpus to the user_features
        nu_features = np.where(
            self.pipe.feature_corpus == 1,
            np.ones(self.pipe.user_features.shape) + self.feat_boost,
            np.ones(self.pipe.user_features.shape)
        )
        self.pipe.user_features = self.pipe.user_features + nu_features
        num_answers = 0
        while num_answers < self.max_answers:
            # Adding the instances
            num_answers += self.pipe.instance_bootstrap(
                self.get_labeled_instance,
                max_iterations=self.cycle_len
            )
            self.pipe._train()
            self.pipe._expectation_maximization()
        self.instance_name = 'inst{}-featb{}'.format(self.instance_number+12,
                                                     self.feat_boost)
        self.save_session()
