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


def get_class(words, classes):
    print "*******************************************************"
    print "\nWhat is the correct template? Write the number or STOP\n"
    question = ' '.join([word.token for word in words])
    print colored(question, "red", "on_white", attrs=["bold"])
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
    pipe.bootstrap(get_class)

    print pipe.get_report()

    filename = raw_input("Insert the filename to save or press enter\n")
    if filename:
        pipe.save_session("sessions/{}".format(filename))

    if args.label_corpus:
        print "Adding {} classes.".format(pipe.label_corpus())


if __name__ == '__main__':
    main()