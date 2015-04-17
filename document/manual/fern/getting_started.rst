Getting started
===============
Unless you obtain one of the prebuilt installation packages for your platform, you need to build Geoneric Fern yourself. In this document we will show you how that is done, it is not very complicated.

First, you must install the prerequisites the Geoneric Fern software depends on, after which you can configure and build Geoneric Fern itself.


Prerequisites
-------------
The Geoneric Fern software depends on 3rd party software. Depending on the Geoneric Fern modules you want to build, you need or need not to install dependencies. Below we list the dependencies and whether or not they are required:

+------------+-----------------------------------------+
| Dependency | Required or not                         |
+============+=========================================+
| Boost      | Always                                  |
+------------+-----------------------------------------+
| NetCDF     | `FERN_BUILD_IO` with `FERN_WITH_NETCDF` |
+------------+-----------------------------------------+
| HDF5       | `FERN_BUILD_IO` with `FERN_WITH_HDF5`   |
+------------+-----------------------------------------+
| GDAL       | `FERN_BUILD_IO` with `FERN_WITH_GDAL`   |
+------------+-----------------------------------------+

Dependencies can be installed using your system's package manager or built. In the latter case you may want to consider using using Geoneric `Peacock`_, which can build all requirements needed by Geoneric Fern for various platforms. The Geoneric Fern sources contains a script (`environment/script/peacock.sh`) which builds all requirements using Peacock.

Configure Geoneric Fern
-----------------------
Geoneric Fern can be built using `CMake`_.

Initially, all configuration options are set to `FALSE`.

`FERN_BUILD_ALL`
    Build all Fern modules.

`FERN_BUILD_ALGORITHM`
    Build Fern.Algorithm module.

`FERN_BUILD_IO`
    Build Fern.IO module.

`FERN_BUILD_PYTHON`
    Build Fern.Python module.


`FERN_BUILD_TEST`
    Build Fern tests.

`FERN_BUILD_DOCUMENTATION`
    Build Fern documentation.


`FERN_WITH_NETCDF`
    Include support for NetCDF in Fern.IO module.

`FERN_WITH_HDF5`
    Include support for HDF5 in Fern.IO module.

`FERN_WITH_GDAL`
    Include support for GDAL in Fern.IO module.


Some CMake configuration options imply the use of other configuration options:

- `FERN_BUILD_ALL` implies `FERN_BUILD_*`.
- `FERN_BUILD_PYTHON` implies `FERN_BUILD_ALGORITHM`, `FERN_BUILD_IO`,
  `FERN_WITH_GDAL`.


Instructions per platform:

.. toctree::
   :maxdepth: 1

   getting_started_on_windows
   getting_started_on_linux
   getting_started_on_os_x


.. _CMake: http://www.cmake.org
.. _Peacock: https://github.com/geoneric/peacock
