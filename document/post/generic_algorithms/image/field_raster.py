#!/usr/bin/env python
import os.path
import sys
import docopt
import matplotlib.pyplot as pyplot
import numpy
import checked_call


usage = \
"""
Create field and raster image

Usage:
  {command} <output_file>
  {command} -h | --help

Arguments:
  <output_file>   Name of output image file.

Options:
  -h --help                        Show this screen.
""".format(command=os.path.split(sys.argv[0])[1])


@checked_call.checked_call
def execute(
        *arguments):
    # Based on: http://matplotlib.org/examples/images_contours_and_fields/interpolation_methods.html

    interpolation_methods = ["lanczos", "none"]

    nr_rows = 15
    nr_cols = 10
    grid = numpy.random.rand(nr_rows, nr_cols)

    titles = ["2D spatial field", "{} x {} raster".format(nr_rows, nr_cols)]

    fig, axes = pyplot.subplots(1, 2, figsize=(6, 3),
        subplot_kw={'xticks': [], 'yticks': []})

    fig.subplots_adjust(hspace=0.3, wspace=0.05)

    for axis, interpolation_method, title in zip(axes.flat,
            interpolation_methods, titles):
        axis.imshow(grid, interpolation=interpolation_method)
        axis.set_title(title)

    pyplot.savefig("field_raster.png", transparent=True)


if __name__ == "__main__":
    arguments = docopt.docopt(usage)
    output_file_pathname = arguments["<output_file>"]
    status = execute(output_file_pathname)
    sys.exit(status)
