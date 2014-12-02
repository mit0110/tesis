
"""Experiment number one

Run the boostrap for features and instances given the user the choice for one
of them after each labeling.
"""

from termcolor import colored
from base_experiment import BaseExperiment
from metrics import (RecognitionVsFeatBoost, AccuracyVsFeatBoost,
                     KappaVsFeatBoost)
import numpy as np


class Experiment12(BaseExperiment):
    def __init__(self, ActivePipeline):
        super(Experiment12, self).__init__(ActivePipeline)
        self.number = 12
        self.metrics = [RecognitionVsFeatBoost(), AccuracyVsFeatBoost(),
                        KappaVsFeatBoost()]
        self.description = ("Feature training with different boost.")
        self.feat_boost_max = 200
        self.instance_number = 92

    def run(self):
        print "Running experiment number {0}: {1}".format(self.number,
                                                          self.description)
        self.pipe = self.pipe_class(emulate=True, **self.experiment_config)
        # Adding the instances
        self.pipe.instance_bootstrap(
            self.get_labeled_instance, max_iterations=self.instance_number
        )
        # Adding the features from the corpus to the user_features
        for boost in range(1, self.feat_boost_max, 10):
            self.pipe.feature_boost = boost
            nu_features = np.where(
                self.pipe.feature_corpus == 1,
                np.ones(self.pipe.user_features.shape) + boost,
                np.ones(self.pipe.user_features.shape)
            )
            self.pipe.user_features = self.pipe.user_features + nu_features
            self.pipe._train()
            self.pipe._expectation_maximization()
        self.instance_name = 'inst{}-featb{}'.format(self.instance_number+12,
                                                     self.feat_boost_max)
        self.save_session()
