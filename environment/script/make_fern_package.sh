#!/usr/bin/env bash
set -e
set -x


function print_usage()
{
    echo -e "\
usage: $0 [-h] <build_prefix> <install_prefix> <fern_source>

-h              Show (this) usage information.

build_prefix    Directory to store intermediate files.
install_prefix  Directory to install the resulting files.
fern_sources    URL to Git repository with Fern sources."
}


# build_prefix
# install_prefix
function parse_commandline()
{
    while getopts h option; do
        case $option in
            h) print_usage; exit 0;;
            *) print_usage; exit 2;;
        esac
    done
    shift $((OPTIND-1))

    if [ $# -ne 3 ]; then
        print_usage
        exit 2
    fi

    build_prefix=`readlink -m $1`
    install_prefix=`readlink -m $2`
    fern_sources=$3
}


# os: Cygwin, GNU/Linux
function determine_platform()
{
    if [ `uname -o 2>/dev/null` ]; then
        os=`uname -o`
    else
        os=`uname`
    fi

    if [ $os == "Cygwin" ]; then
        ### # Path to ml64.exe, required by Boost.Context.
        ### vs_2008_root=`cygpath "$VS90COMNTOOLS"`
        ### amd64_root="$vs_2008_root/../../VC/BIN/amd64"

        ### # Path to compiler.
        ### mingw_root=/cygdrive/c/mingw64/bin

        ### export PATH="$mingw_root:$amd64_root:$PATH"

        make=mingw32-make
    else
        make=make
    fi


    cmake=cmake
    cmake_generator="Unix Makefiles"
    find=/usr/bin/find
    cmake_make_program=$make
    wget=wget
    unzip=unzip
    sed="sed -i.tmp"  # Make sure this is GNU sed!
}


function print_summary()
{
    echo -e "\
build prefix       : $build_prefix
install_prefix     : $install_prefix
Operating system   : $os"
}


function ask_permission_to_continue()
{
    echo -e "\
Continue building? [y or n]: "

    local answer
    read answer

    if [ "$answer" != "y" ]; then
       exit 0
    fi

    mkdir -p $build_prefix
    mkdir -p $install_prefix
}


function configure_versions()
{
    # fern_version=head
    fern_version=`date +"%Y%m%d"`
    fern_install_prefix=$install_prefix/fern-$fern_version
}


function native_path()
{
    local pathname=$1
    local variable_name=$2
    local native_pathname=$pathname

    if [ $os == "Cygwin" ]; then
        native_pathname=`cygpath -m $native_pathname`
    fi

    eval $variable_name="$native_pathname"
}


function rebuild_using_cmake()
{
    local source_directory=$1
    local binary_directory=$2
    local install_prefix=$3
    local build_type=$4
    local options=$5

    native_path $source_directory native_source_directory
    native_path $install_prefix native_install_prefix

    rm -fr $binary_directory/*

    mkdir -p $binary_directory
    cd $binary_directory
    $cmake \
        -DCMAKE_BUILD_TYPE=$build_type \
        -G"$cmake_generator" \
        -DCMAKE_MAKE_PROGRAM=$cmake_make_program \
        -DCMAKE_INSTALL_PREFIX="$native_install_prefix" \
        $options \
        $native_source_directory
    $cmake --build . --config $build_type
    $cmake --build . --config $build_type --target install
}


function build_fern()
{
    cd $build_prefix

    if [ ! -d fern_sources ]; then
        rm -fr fern_sources
        git clone $fern_sources fern_sources
    else
        cd fern_sources
        git pull
        cd ..
    fi

    rm -fr $fern_install_prefix

    function build()
    {
        local build_type=$1
        local options="
            -DFERN_ALGORITHM:BOOL=TRUE
        "

        ### if [ $os == "Cygwin" ]; then
        ###     options="
        ###         $options
        ###         -DBOOST_ROOT=`cygpath -m $boost_install_prefix`
        ###     "
        ### fi

        rebuild_using_cmake \
            $build_prefix/fern_sources \
            $build_prefix/fern_objects \
            $fern_install_prefix \
            $build_type \
            "$options"

        native_path $build_prefix/fern_objects native_binary_directory
        $cmake \
            --build $native_binary_directory \
            --config $build_type \
            --target test
    }

    rm -fr $fern_install_prefix/*

    if [ $os == "Cygwin" ]; then
        build Debug
    fi

    build Release
}


function create_fern_zip()
{
    install_prefix_basename=`basename $fern_install_prefix`
    zip_filename=${install_prefix_basename}.zip

    cd $install_prefix
    rm -f $zip_filename
    zip -r -q -9 $zip_filename $install_prefix_basename
    mv $zip_filename ..
    cd ..
    echo `pwd`/$zip_filename
}


parse_commandline $*
determine_platform
print_summary
ask_permission_to_continue
configure_versions
mkdir -p $build_prefix $install_prefix

build_fern
create_fern_zip
