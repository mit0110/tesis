"""
Usage:
    run_experiment.py <experiment_number> [<args> ...]

There must be a corresponding class named ExperimentX (where X is the
experiment number) in the module experiments. Each experiment must have an
init function that takes as first argument the ActivePipeline class and a
method called run. It is recommended to use a subclass of BaseExperiment.
"""

import experiments
from activepipe import ActivePipeline
from docopt import docopt

if __name__ == '__main__':
    opts = docopt(__doc__, version=0.1)
    experiment_number = opts['<experiment_number>']
    args = opts['<args>']
    exp_class = getattr(experiments, 'Experiment{}'.format(experiment_number))
    experiment = exp_class(ActivePipeline, *args)
    experiment.run()
