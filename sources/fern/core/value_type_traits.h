#pragma once
#include "fern/core/string.h"
#include "fern/core/value_type.h"


namespace fern {

template<
    ValueType value_type>
struct ValueTypeTraits
{
};


template<>
struct ValueTypeTraits<VT_UINT8>
{
    typedef uint8_t type;
};


template<>
struct ValueTypeTraits<VT_INT8>
{
    typedef int8_t type;
};


template<>
struct ValueTypeTraits<VT_UINT16>
{
    typedef uint16_t type;
};


template<>
struct ValueTypeTraits<VT_INT16>
{
    typedef int16_t type;
};


template<>
struct ValueTypeTraits<VT_UINT32>
{
    typedef uint32_t type;
};


template<>
struct ValueTypeTraits<VT_INT32>
{
    typedef int32_t type;
};


template<>
struct ValueTypeTraits<VT_UINT64>
{
    typedef uint64_t type;
};


template<>
struct ValueTypeTraits<VT_INT64>
{
    typedef int64_t type;
};


template<>
struct ValueTypeTraits<VT_FLOAT32>
{
    typedef float type;
};


template<>
struct ValueTypeTraits<VT_FLOAT64>
{
    typedef double type;
};


template<>
struct ValueTypeTraits<VT_STRING>
{
    typedef String type;
};

} // namespace fern
