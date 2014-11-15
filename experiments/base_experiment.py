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
    def __init__(self, ActivePipeline, *args):
        self.pipe_class = ActivePipeline
        self.args = args
        self.get_labeled_instance = get_labeled_instance
        self.get_class = get_class
        self.get_labeled_features = get_labeled_features
        self.experiment_config = {}

    def run(self):
        raise NotImplementedError

