Getting started
===============
Unless you obtained a prebuilt installation packages for your platform, you need to build Geoneric Fern yourself. In this document we will show you how that is done, it is not very complicated.

The steps to perform are:

#. Obtain the source code.
#. Configure Fern.
#. Build Fern.
#. Install Fern.

In case you have the prerequisites Fern depends on installed on your machine, this comes down to this sequence of commands:

::

   git clone --recursive https://github.com/geoneric/fern.git
   mkdir fern_build && cd fern_build
   cmake ../fern
   cmake --build . --target all
   cmake --build . --target install

This will not build anything useful for you, though. You must select the components you want to build, by passing configuration variables to CMake see :ref:`configure_fern`).

In case you don't have all prerequites installed on your machine, you must make sure you get them installed.


Prerequisites
-------------
The Fern software depends on 3rd party software. Depending on the Fern modules you want to build, you need or need not to install dependencies. Below we list the dependencies and whether or not they are required. The ``FERN_`` symbols correspond with variables that can be passed to CMake when configuring a Fern build (see :ref:`configure_fern`).

+------------+------------------------------------------------+
| Dependency | Required or not                                |
+============+================================================+
| Boost      | Always                                         |
+------------+------------------------------------------------+
| NetCDF     | ``FERN_BUILD_IO`` with ``FERN_IO_WITH_NETCDF`` |
+------------+------------------------------------------------+
| HDF5       | ``FERN_BUILD_IO`` with ``FERN_IO_WITH_HDF5``   |
+------------+------------------------------------------------+
| GDAL       | ``FERN_BUILD_IO`` with ``FERN_IO_WITH_GDAL``   |
+------------+------------------------------------------------+

Dependencies can be installed using your system's package manager or built. In the latter case you may want to consider using using Geoneric `Peacock`_, which can build all requirements needed by Fern for various platforms. The Fern sources contains a script (`environment/script/peacock.sh`) which builds all requirements using Peacock.


.. _configure_fern:

Configure Fern
--------------
Fern can be built using `CMake`_. Passing configuration options to CMake works likes this:

::

    cmake -D<option> ...

Initially, all configuration options are set to `FALSE`.

`FERN_BUILD_ALL`
    Build all Fern modules.

`FERN_BUILD_ALGORITHM`
    Build Fern.Algorithm module.

`FERN_BUILD_IO`
    Build Fern.IO module.

`FERN_BUILD_LANGUAGE`
    Build Fern.Language module.

`FERN_BUILD_PYTHON`
    Build Fern.Python module.


`FERN_BUILD_TEST`
    Build Fern tests.

`FERN_BUILD_DOCUMENTATION`
    Build Fern documentation.

`FERN_IO_WITH_ALL`
    Include support for all supported formats in Fern.IO module.

`FERN_IO_WITH_GDAL`
    Include support for GDAL in Fern.IO module.

`FERN_IO_WITH_GPX`
    Include support for GPX in Fern.IO module.

`FERN_IO_WITH_HDF5`
    Include support for HDF5 in Fern.IO module.

`FERN_IO_WITH_NETCDF`
    Include support for NetCDF in Fern.IO module.


Some CMake configuration options imply the use of other configuration options:

- `FERN_BUILD_ALL` implies `FERN_BUILD_*`.
- `FERN_BUILD_PYTHON` implies `FERN_BUILD_ALGORITHM`, `FERN_BUILD_IO`,
  `FERN_IO_WITH_GDAL`.
- `FERN_BUILD_LANGUAGE` implies `FERN_BUILD_ALGORITHM`, `FERN_BUILD_IO`,
  `FERN_IO_WITH_GDAL`, `FERN_IO_WITH_HDF5`.
- `FERN_IO_WITH_ALL` implies `FERN_IO_WITH_*`.


An example of configuring a Fern build on Linux where CMake is requested to

- Generate Unix Makefiles
- Configure for a release build (this is the default)
- Configure to build only the Fern.Algorithm library
- Install the results in a specific location

::

   cmake \
       -G"Unix Makefiles" \
       -DCMAKE_BUILD_TYPE=Release \
       -DFERN_BUILD_ALGORITHM:BOOL=TRUE \
       -DCMAKE_MAKE_PROGRAM=make \
       -DCMAKE_INSTALL_PREFIX=/opt/fern \
       $HOME/tmp/fern


.. _CMake: http://www.cmake.org
.. _Peacock: https://github.com/geoneric/peacock
