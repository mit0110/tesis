"""
Interfaz between Quepy and ACTIVEpipe.

Usage:
    -- For classification
    -- For training
"""

import argparse
from termcolor import colored

from activepipe import ActivePipeline
from feature_extraction import get_features


def get_class(instance, classes):
    print classes
    print "*******************************************************"
    print "\nWhat is the correct template? Write the number or STOP\n"
    print colored(instance, "red", "on_white", attrs=["bold"])
    message = "{} - {}"
    for (counter, class_name) in enumerate(classes):
        print message.format(counter, class_name)
    line = raw_input(">>> ")
    if line.lower() == 'stop':
        return 'stop'

    try:
        line = int(line)
        prediction = classes[line]
    except (ValueError, IndexError):
        print colored("Please insert one of the listed numbers", "red")
        return 'stop'
    print colored("Adding result", "green")
    return prediction


def get_class(classes_list):
    print "*******************************************************"
    print "Choose a class number to label its features"
    for index, class_ in enumerate(classes_list):
        print index, class_
    line = raw_input(">>> ")
    try:
        index = int(index)
    except ValueError:
        print 'Choose a number'
        return False
    if index < 0 or index >= len(classes_list):
        print 'Choose a number between 0 and ', len(classes_list)
        return False
    return classes_list[index]


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


config = {
    'features': get_features(),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--output_file', required=False,
                        type=str)
    parser.add_argument('-e', '--emulate', action='store_true')
    parser.add_argument('-l', '--label_corpus', action='store_true')
    args = parser.parse_args()

    pipe = ActivePipeline(session_filename=args.output_file,
                          emulate=args.emulate, **config)
    # pipe.instance_bootstrap(get_class)
    pipe.feature_bootstrap(get_class, get_labeled_features)

    print pipe.get_report()

    filename = raw_input("Insert the filename to save or press enter\n")
    if filename:
        pipe.save_session("sessions/{}".format(filename))

    if args.label_corpus:
        print "Adding {} classes.".format(pipe.label_corpus())


if __name__ == '__main__':
    main()