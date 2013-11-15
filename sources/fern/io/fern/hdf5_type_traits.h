#pragma once
#include <cpp/H5Cpp.h>


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
