// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <string>
#include <hdf5.h>


namespace fern {
namespace language {

template<
    H5T_class_t TypeClass>
struct HDF5TypeClassTraits
{
    static std::string const name;
};


template<>
struct HDF5TypeClassTraits<H5T_INTEGER>
{
    static std::string const name;
};


template<>
struct HDF5TypeClassTraits<H5T_FLOAT>
{
    static std::string const name;
};


template<>
struct HDF5TypeClassTraits<H5T_TIME>
{
    static std::string const name;
};


template<>
struct HDF5TypeClassTraits<H5T_STRING>
{
    static std::string const name;
};


template<>
struct HDF5TypeClassTraits<H5T_NO_CLASS>
{
    static std::string const name;
};


template<>
struct HDF5TypeClassTraits<H5T_BITFIELD>
{
    static std::string const name;
};


template<>
struct HDF5TypeClassTraits<H5T_OPAQUE>
{
    static std::string const name;
};


template<>
struct HDF5TypeClassTraits<H5T_COMPOUND>
{
    static std::string const name;
};


template<>
struct HDF5TypeClassTraits<H5T_REFERENCE>
{
    static std::string const name;
};


template<>
struct HDF5TypeClassTraits<H5T_ENUM>
{
    static std::string const name;
};


template<>
struct HDF5TypeClassTraits<H5T_VLEN>
{
    static std::string const name;
};


template<>
struct HDF5TypeClassTraits<H5T_ARRAY>
{
    static std::string const name;
};


template<>
struct HDF5TypeClassTraits<H5T_NCLASSES>
{
    static std::string const name;
};

} // namespace language
} // namespace fern
