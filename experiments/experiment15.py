
"""Experiment number one

Run the boostrap for features and instances given the user the choice for one
of them after each labeling.
"""

from termcolor import colored
from base_experiment import BaseExperiment
from metrics import (RecognitionVsFeatBoost, AccuracyVsFeatBoost,
                     KappaVsFeatBoost)
import numpy as np


class Experiment15(BaseExperiment):
    def __init__(self, ActivePipeline):
        super(Experiment15, self).__init__(ActivePipeline)
        self.number = 15
        self.description = ("Active learning on features with no user"
                            "using minimum IG.")
        self.feat_boost = 100
        self.cycle_len = 1

    def run(self):
        print "Running experiment number {0}: {1}".format(self.number,
                                                          self.description)
        self.pipe = self.pipe_class(emulate=True, **self.experiment_config)
        self.pipe.instance_bootstrap(
            self.get_labeled_instance,
            max_iterations=self.max_answers
        )
        num_answers = 0
        possible_features = [i for i in
            range(self.pipe.classifier.feat_information_gain.shape[0])
            if np.any(self.pipe.feature_corpus.T[i] > 0)
        ]
        n_classes = len(self.pipe.classes)
        print 'possible_features', len(possible_features)
        while possible_features:
            # Select the new feature
            new_f = min(possible_features,
                key=lambda x: self.pipe.classifier.feat_information_gain[x]
            )
            # Add the new feature
            new_f_v = np.where(self.pipe.feature_corpus.T[new_f] > 0,
                               np.ones(n_classes) + self.feat_boost,
                               np.ones(n_classes))
            self.pipe.user_features.T[new_f] = new_f_v

            self.pipe.new_features += (self.pipe.feature_corpus.T[new_f] > 0).sum()
            possible_features.remove(new_f)
            print ("Adding feature", new_f,
                   self.pipe.classifier.feat_information_gain[new_f])

            self.pipe._train()
            self.pipe._expectation_maximization()
        self.instance_name = 'inst{}-featb{}-{}'.format(
            self.max_answers+12, self.feat_boost, self.max_feat_answers
        )
        self.save_session()
