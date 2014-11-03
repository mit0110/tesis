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


def get_features_4_class(class_name, features):
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
    pipe.feature_bootstrap(get_features_4_class)

    print pipe.get_report()

    filename = raw_input("Insert the filename to save or press enter\n")
    if filename:
        pipe.save_session("sessions/{}".format(filename))

    if args.label_corpus:
        print "Adding {} classes.".format(pipe.label_corpus())


if __name__ == '__main__':
    main()