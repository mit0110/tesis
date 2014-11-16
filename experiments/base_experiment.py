"""
BaseExperiment class and default functions to run the activepipe bootstraps.
"""
from termcolor import colored


def get_labeled_instance(instance, classes):
    """Propose to the user an instance and a list of classes to label.

    Params:
        instance: the representation in natural languange of an instance to be
        labeled by the user.
        classes: a list of classes names in order as possible labels for the
        instance.

    Returns:
        A string. One of the classes names listed in the argument classes,
        stop to finish the bootstrap or train to retrain the classifier.
    """
    print "*******************************************************"
    print "\nWhat is the correct template? Write the number or STOP\n"
    print colored(instance, "red", "on_white", attrs=["bold"])
    message = "{} - {}"
    for (counter, class_name) in enumerate(classes):
        print message.format(counter, class_name)
    line = raw_input(">>> ")
    try:
        line = int(line)
        prediction = classes[line]
    except (ValueError, IndexError):
        return line
    print colored("Adding result", "green")
    return prediction


def get_class(classes_list):
    print "*******************************************************"
    print "Choose a class number to label its features"
    for index, class_ in enumerate(classes_list):
        print index, class_
    line = raw_input(">>> ")
    try:
        line = int(line)
    except ValueError:
        print 'Choose a number'
        return line if (line == 'stop' or line == 'train') else None
    if line < 0 or line >= len(classes_list):
        print 'Choose a number between 0 and ', len(classes_list)
        return False
    return classes_list[line]


def get_labeled_features(class_name, features):
    print "*******************************************************"
    print "Insert the asociated feature numbers separated by a blank space."
    print class_name
    for number, feature in enumerate(features):
        print number, feature
    line = raw_input(">>> ")
    if line.lower() == 'stop':
        return 'stop'
    indexes = line.split()
    result = []
    for index in indexes:
        try:
            index = int(index)
            result.append(features[index])
        except ValueError:
            pass
    return result


class BaseExperiment(object):
    """A base class to define experiments.

    A new experiment must define the method run and the metrics as minimum.

    Attributes:
        pipe_class: The class to instanciate the active learn pipe.
        get_labeled_instance: A function. Can be used as a parameter for the
        instance bootstrap of the active pipe.
        get_class: A function. Can be used as a parameter for the
        feature bootstrap of the active pipe.
        get_labeled_features: A function. Can be used as a parameter for the
        feature bootstrap of the active pipe.
        experiment_config: A dictionary with the default configuration to
        run an experiment.
        metrics: A list with Metric instances that must be obteined after
        running the experiment from the session file.

    """

    def __init__(self, ActivePipeline):
        """Sets the default attributes for an experiment.

        Args:
            ActivePipeline: The class for the active learn pipe that will be
            tested.
        """
        self.pipe_class = ActivePipeline
        self.get_labeled_instance = get_labeled_instance
        self.get_class = get_class
        self.get_labeled_features = get_labeled_features
        self.experiment_config = {}
        self.metrics = []
        self.number = 0

    def run(self):
        """Runs the bootstrap cicles of the ActivePipeline."""
        raise NotImplementedError

    def get_metrics(self):
        """Extracts the metrics of self.metrics from the session file.

        Saves the results into results/experimentN/self.results_filename
        """
        filename_base = 'experiments/results/experiment{}/'.format(self.number)
        for metric in self.metrics:
            metric.get_from_session_file(self.session_filename)
            metric.store_in_file(filename_base + metric.name)
