"""
Interfaz between Quepy and ACTIVEpipe.

Usage:
    -- For classification
    -- For training
"""

import argparse

from activepipe import ActivePipeline



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--output_file', required=False,
                        type=str)
    parser.add_argument('-e', '--emulate', action='store_true')
    parser.add_argument('-l', '--label_corpus', action='store_true')
    args = parser.parse_args()
    pipe = ActivePipeline(session_filename=args.output_file,
                          emulate=args.emulate,
                          label_corpus=args.label_corpus)
    pipe.bootstrap()


if __name__ == '__main__':
    main()