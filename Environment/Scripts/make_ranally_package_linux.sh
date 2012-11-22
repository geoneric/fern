#!/usr/bin/env bash
set -e

build_type="Release"
base_name="ranally-`date +%Y%m%d`"
build_root=`pwd`
install_prefix=`pwd`/$base_name

make_ranally_package.sh "$build_root" "$install_prefix" $build_type
tar zcf $base_name.tar.gz $base_name
ls -lh $base_name.tar.gz
