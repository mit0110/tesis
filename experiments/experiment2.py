
"""Experiment number one

Run the boostrap for features and instances given the user the choice for one
of them after each labeling.
"""
from base_experiment import BaseExperiment


class Experiment2(BaseExperiment):
    def __init__(self, ActivePipeline):
        super(Experiment2, self).__init__(ActivePipeline)
        self.number = 2
        self.description = ("Active Learning on instaces using entropy.")
        self.cycle_len = 1

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
#            self.pipe._expectation_maximization()
        self.get_name()
        self.save_session()

