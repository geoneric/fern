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
struct ValueTypeTraits<VT_BOOL>
{
    using type = bool;
};


template<>
struct ValueTypeTraits<VT_UINT8>
{
    using type = uint8_t;
};


template<>
struct ValueTypeTraits<VT_INT8>
{
    using type = int8_t;
};


template<>
struct ValueTypeTraits<VT_UINT16>
{
    using type = uint16_t;
};


template<>
struct ValueTypeTraits<VT_INT16>
{
    using type = int16_t;
};


template<>
struct ValueTypeTraits<VT_UINT32>
{
    using type = uint32_t;
};


template<>
struct ValueTypeTraits<VT_INT32>
{
    using type = int32_t;
};


template<>
struct ValueTypeTraits<VT_UINT64>
{
    using type = uint64_t;
};


template<>
struct ValueTypeTraits<VT_INT64>
{
    using type = int64_t;
};


template<>
struct ValueTypeTraits<VT_FLOAT32>
{
    using type = float;
};


template<>
struct ValueTypeTraits<VT_FLOAT64>
{
    using type = double;
};


template<>
struct ValueTypeTraits<VT_STRING>
{
    using type = String;
};

} // namespace fern
