#!/usr/bin/env python
"""Rename project.

Usage:
  rename_project.py <old_name> <new_name>
  rename_project.py --help

Options:
  --help  Show this screen.
"""
import sys
import docopt
import fern


@fern.checked_call
def rename_project(
        model_filename,
        prefix):
    fern.rename_project(model_filename, prefix)


if __name__ == "__main__":
    arguments = docopt.docopt(__doc__)
    old_name = arguments["<old_name>"]
    new_name = arguments["<new_name>"]

    sys.exit(rename_project(old_name, new_name))
