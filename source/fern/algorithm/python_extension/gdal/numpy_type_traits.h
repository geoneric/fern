#pragma once


namespace fern {

template<
    class T>
struct NumpyTypeTraits
{
};


template<>
struct NumpyTypeTraits<uint8_t>
{
    static int const data_type = NPY_UINT8;
};


template<>
struct NumpyTypeTraits<uint16_t>
{
    static int const data_type = NPY_UINT16;
};


template<>
struct NumpyTypeTraits<int16_t>
{
    static int const data_type = NPY_INT16;
};


template<>
struct NumpyTypeTraits<uint32_t>
{
    static int const data_type = NPY_UINT32;
};


template<>
struct NumpyTypeTraits<int32_t>
{
    static int const data_type = NPY_INT32;
};


template<>
struct NumpyTypeTraits<uint64_t>
{
    static int const data_type = NPY_UINT64;
};


template<>
struct NumpyTypeTraits<int64_t>
{
    static int const data_type = NPY_INT64;
};


template<>
struct NumpyTypeTraits<float>
{
    static int const data_type = NPY_FLOAT32;
};


template<>
struct NumpyTypeTraits<double>
{
    static int const data_type = NPY_FLOAT64;
};

} // namespace fern
