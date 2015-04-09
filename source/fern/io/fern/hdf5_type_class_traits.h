// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <H5Cpp.h>
#include "fern/core/string.h"


namespace fern {

template<
    H5T_class_t TypeClass>
struct HDF5TypeClassTraits
{
    static String const name;
};


template<>
struct HDF5TypeClassTraits<H5T_INTEGER>
{
    static String const name;
};


template<>
struct HDF5TypeClassTraits<H5T_FLOAT>
{
    static String const name;
};


template<>
struct HDF5TypeClassTraits<H5T_TIME>
{
    static String const name;
};


template<>
struct HDF5TypeClassTraits<H5T_STRING>
{
    static String const name;
};


template<>
struct HDF5TypeClassTraits<H5T_NO_CLASS>
{
    static String const name;
};


template<>
struct HDF5TypeClassTraits<H5T_BITFIELD>
{
    static String const name;
};


template<>
struct HDF5TypeClassTraits<H5T_OPAQUE>
{
    static String const name;
};


template<>
struct HDF5TypeClassTraits<H5T_COMPOUND>
{
    static String const name;
};


template<>
struct HDF5TypeClassTraits<H5T_REFERENCE>
{
    static String const name;
};


template<>
struct HDF5TypeClassTraits<H5T_ENUM>
{
    static String const name;
};


template<>
struct HDF5TypeClassTraits<H5T_VLEN>
{
    static String const name;
};


template<>
struct HDF5TypeClassTraits<H5T_ARRAY>
{
    static String const name;
};


template<>
struct HDF5TypeClassTraits<H5T_NCLASSES>
{
    static String const name;
};

} // namespace fern
