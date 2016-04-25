Fern
====
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


Supported platforms
-------------------
Fern is portable software. The following table lists the platforms that we use ourselves to verify whether Fern builds and works (Fern.Algorithm module only for now). Given this range in platforms, we feel confident that Fern builds (or can be made to build) on other platforms as well.

| OS | Compilers | Status |
|----|-----------|--------|
| Linux | gcc-4.9 · x64 | [![Linux build Status](https://travis-ci.org/geoneric/fern.svg?branch=master)](https://travis-ci.org/geoneric/fern)  |
| Linux | clang-3.7 · x64 | Checked 'by hand' |
| Windows | vs-2015 · x64 | [![Windows build Status](https://ci.appveyor.com/api/projects/status/github/geoneric/fern?branch=master&svg=true)](https://ci.appveyor.com/project/kordejong/fern) |
| OS X | clang-3.7 · x64 | Checked 'by hand' |

20160425: Windows build fails because of [bug in CMake 3.5](https://cmake.org/Bug/view.php?id=16020) which is currently used by AppVeyor.
