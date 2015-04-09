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


namespace fern {

template<
    class T>
struct HDF5TypeTraits
{
    static H5::PredType const data_type;
    static H5T_class_t const type_class;
};


template<>
struct HDF5TypeTraits<uint8_t>
{
    static H5::PredType const data_type;
    static H5T_class_t const type_class = H5T_INTEGER;
};


template<>
struct HDF5TypeTraits<uint16_t>
{
    static H5::PredType const data_type;
    static H5T_class_t const type_class = H5T_INTEGER;
};


template<>
struct HDF5TypeTraits<uint32_t>
{
    static H5::PredType const data_type;
    static H5T_class_t const type_class = H5T_INTEGER;
};


template<>
struct HDF5TypeTraits<uint64_t>
{
    static H5::PredType const data_type;
    static H5T_class_t const type_class = H5T_INTEGER;
};


template<>
struct HDF5TypeTraits<int8_t>
{
    static H5::PredType const data_type;
    static H5T_class_t const type_class = H5T_INTEGER;
};


template<>
struct HDF5TypeTraits<int16_t>
{
    static H5::PredType const data_type;
    static H5T_class_t const type_class = H5T_INTEGER;
};


template<>
struct HDF5TypeTraits<int32_t>
{
    static H5::PredType const data_type;
    static H5T_class_t const type_class = H5T_INTEGER;
};


template<>
struct HDF5TypeTraits<int64_t>
{
    static H5::PredType const data_type;
    static H5T_class_t const type_class = H5T_INTEGER;
};


template<>
struct HDF5TypeTraits<float>
{
    static H5::PredType const data_type;
    static H5T_class_t const type_class = H5T_FLOAT;
};


template<>
struct HDF5TypeTraits<double>
{
    static H5::PredType const data_type;
    static H5T_class_t const type_class = H5T_FLOAT;
};

} // namespace fern
