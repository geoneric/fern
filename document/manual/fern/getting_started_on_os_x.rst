Getting started on OS X
-----------------------
The process is as folows:

#. Unpack the source code.
#. Configure the build.
#. Build.
#. Install.


::

   tar -zxf fern-<version>.tar.gz
   cd fern-<version>
   cmake .
   cmake --build . --target install
