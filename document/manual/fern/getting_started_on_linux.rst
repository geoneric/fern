Getting started on Linux
========================
The process is as folows:

In case all prerequisites are installed and configured, installing Fern is matter of performing these steps:

#. Unpack the source code.
#. Configure.
#. Build.
#. Install.


::

   tar -zxf fern-<version>.tar.gz
   cd fern-<version>
   cmake --build . --config Release --target all
   cmake --build . --config Release --target test
   cmake --build . --config Release --target install


An example where CMake is requested to

- Generate Unix Makefiles
- Configure for a release build
- Configure to build only the Fern.Algorithm library
- Install the results in a specific location

::

   cmake \
       -G"Unix Makefiles" \
       -DCMAKE_BUILD_TYPE=Release \
       -DFERN_ALGORITHM:BOOL=TRUE \
       -DCMAKE_MAKE_PROGRAM=make \
       -DCMAKE_INSTALL_PREFIX=/opt/fern-<version> \
       $HOME/tmp/fern-<version>
