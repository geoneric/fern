#!/usr/bin/env bash
set -e

build_type="Develop"

# Fern project. ----------------------------------------------------------------
fern_source_directory=$FERN
fern_binary_directory=/tmp/fern_objects
fern_install_prefix=/tmp/fern
fern_external_prefix=$PCRTEAM_PLATFORM

# Configure.
rm -fr $fern_binary_directory
mkdir $fern_binary_directory
cd $fern_binary_directory
cmake \
    -DCMAKE_BUILD_TYPE=$build_type \
    -DCMAKE_INSTALL_PREFIX=$fern_install_prefix \
    $fern_source_directory

# Build.
cd $fern_binary_directory
cmake --build $fern_binary_directory --target all --config $build_type

# Install.
rm -fr $fern_install_prefix
cd $fern_binary_directory
cmake --build $fern_binary_directory --target install --config $build_type
fixup.py $fern_install_prefix $fern_external_prefix

# Model project. ---------------------------------------------------------------
model_basename=bla
model_pathname=/tmp/$model_basename.mod
model_install_prefix=/tmp/$model_basename

# Model.
echo "a = abs(b)" > $model_pathname

# Model binaries.
rm -fr $model_install_prefix

# Make sure stuff installed above is used.
export CMAKE_MODULE_PATH="$CMAKE_MODULE_PATH;$fern_install_prefix/share/cmake"
export PATH="$fern_install_prefix/bin:$PATH"

export CXXFLAGS="-I$fern_install_prefix/include"
export LDFLAGS="-L$fern_install_prefix/lib"
model_to_executable.py $model_pathname $model_install_prefix
cp $fern_install_prefix/lib/*fernlib*.so $model_install_prefix/lib
fixup.py $model_install_prefix $fern_external_prefix

# Sanity test.
$model_install_prefix/bin/$model_basename
