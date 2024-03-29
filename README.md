Fern
====

[![Linux build status](https://github.com/geoneric/fern/workflows/Linux%20CI/badge.svg)](https://github.com/geoneric/fern/actions/workflows/linux.yml)
[![macOS build status](https://github.com/geoneric/fern/workflows/macOS%20CI/badge.svg)](https://github.com/geoneric/fern/actions/workflows/macos.yml)
[![Windows build status](https://github.com/geoneric/fern/workflows/Windows%20CI/badge.svg)](https://github.com/geoneric/fern/actions/workflows/windows.yml)

Software components for scalable geocomputing
---------------------------------------------

Fern is a set of software components targeted at scalable geocomputing. One major component is the Fern.Algorithm library, containing configurable generic algorithms that scale to all processors in a shared memory architecture. As an example of what this might mean to you is that using Fern.Algorithm, you can calculate the slope of a digital elevation model, where:

- The digital elevation model is stored in your favorite data structure.
- The resulting slope values are stored in your favorite data structure.
- You pick the value types of the cells (size of the floating point type in case of the slope algorithm).
- All CPU cores in your machine are used during the calculations.
- You get to decide how the algorithm should handle:
    - No-data in the input.
    - Out of domain values in the input.
    - Out of range values in the output.
    - No-data in the output.
- You don't pay (performance-wise) for features that you don't use.
- The result is calculated at least as fast as a specifically crafted handwritten algorithm would.

Documentation can be found in the [source tree](document/manual/fern) and in our [blog](http://blog.geoneric.eu).

We have not created an official release yet. Nevertheless, Fern is currently being used in the following projects:

- openLISEM (http://blogs.itc.nl/lisem): A spatial model for runoff, floods
  and erosion
- PCRaster (http://www.pcraster.eu): Software for environmental modelling

If you are also a Fern user, then please let us know (info@geoneric.eu), and we will add you to the list. Also, in case you have questions, don't hesitate to contact us.
