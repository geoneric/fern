#!/usr/bin/env bash
set -e

model_basename=bla
model_pathname=/tmp/$model_basename.mod
install_prefix=/tmp/$model_basename

make -C $OBJECTS/fern all
echo "a = abs(b)" > $model_pathname
rm -fr $install_prefix

export CMAKE_MODULE_PATH="$CMAKE_MODULE_PATH;$FERN/environment/cmake"
export CXXFLAGS="-I$FERN/sources"
export LDFLAGS="-L$OBJECTS/fern/bin"
model_to_executable.py $model_pathname $install_prefix

export LD_LIBRARY_PATH=$install_prefix/bin
$install_prefix/bin/$model_basename
