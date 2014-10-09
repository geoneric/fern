#!/usr/bin/env python
"""Create a native executable from a model.

Usage:
  model_to_executable.py <model> <prefix>
  model_to_executable.py --help

Options:
  --help  Show this screen.
  model   Model to compile.
  prefix  Root of directory to store the results in (<prefix>/bin).
"""
import sys
import docopt
import fern


@fern.checked_call
def model_to_executable(
        model_filename,
        prefix):
    fern.model_to_executable(model_filename, prefix)


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)
    model_filename = arguments["<model>"]
    prefix = arguments["<prefix>"]

    sys.exit(model_to_executable(model_filename, prefix))
