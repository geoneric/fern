#!/usr/bin/env bash
set -e


function print_usage()
{
    echo -e "\
usage: $0 [-h] <download_dir> <prefix> <source>

-h              Show (this) usage information.

download_dir    Directory to store downloaded files.
prefix          Directory to install the resulting files.
source          Directory of Peacock sources."
}


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

    download_dir=$1
    prefix=$2
    source=$3
}


function build_peacock()
{
    # Don't build any of the support libraries...
    build_boost=false
    build_hdf5=false
    build_netcdf=false
    build_gdal=false
    build_hpx=false
    build_google_benchmark=false


    # ... except for these machines
    hostname=`hostname`

    if [ $hostname == "sonic.geo.uu.nl" ]; then
        build_boost=true
    fi


    options+=("-Dpeacock_download_dir=$download_dir")
    options+=("-Dpeacock_prefix=$prefix")
    # options+=("-DCMAKE_VERBOSE_MAKEFILE=ON")


    if [ "$build_boost" = true ]; then
        options+=("-Dbuild_boost=true")
        options+=("-Dboost_version=1.57.0")
        options+=("-Dboost_build_boost_atomic=true")
        options+=("-Dboost_build_boost_chrono=true")
        options+=("-Dboost_build_boost_date_time=true")
        options+=("-Dboost_build_boost_filesystem=true")
        options+=("-Dboost_build_boost_program_options=true")
        # Doesn't work icw clang/linux
        # options+=("-Dboost_build_boost_python=true")
        options+=("-Dboost_build_boost_regex=true")
        options+=("-Dboost_build_boost_serialization=true")
        options+=("-Dboost_build_boost_system=true")
        options+=("-Dboost_build_boost_test=true")
        options+=("-Dboost_build_boost_thread=true")
        options+=("-Dboost_build_boost_timer=true")
    fi


    if [ "$build_hdf5" = true ]; then
        # Fern I/O code uses deprecated public API symbols.
        options+=("-Dbuild_hdf5=true")
        options+=("-Dhdf5_cpp_lib=false")
        options+=("-Dhdf5_deprecated_symbols=true")
        options+=("-Dhdf5_parallel=false")
        options+=("-Dhdf5_thread_safe=true")
        options+=("-Dhdf5_version=1.8.14")
    fi


    if [ "$build_netcdf" = true ]; then
        options+=("-Dbuild_netcdf=true")
        options+=("-Dnetcdf_version=4.3.3.1")
    fi


    if [ "$build_hpx" = true ]; then
        # When building HPX, HDF5 must be build with thread_safe=true.
        # This, in turn, prevents support for HDF5 C++ support.
        # HPX.
        options+=("-Dbuild_hpx=true")
        options+=("-Dhpx_version=0.9.10")
    fi


    if [ "$build_gdal" = true ]; then
        options+=("-Dbuild_gdal=true")
        options+=("-Dgdal_version=1.11.1")
        options+=("-Dgdal_build_python_package=true")
        options+=("-Dgdal_with_hdf5=true")
        options+=("-Dgdal_with_netcdf=true")
    fi


    if [ "$build_google_benchmark" = true ]; then
        options+=("-Dbuild_google_benchmark=true")
        options+=("-Dgoogle_benchmark_version=1.3.0")
    fi


    cmake "${options[@]}" $source
    cmake --build . --target all
}


parse_commandline $*
build_peacock
