#pragma once
#include "geoneric/core/value_types.h"


namespace geoneric {

template<
    class T>
struct TypeTraits
{
    // static ValueType const value_type;

    // static ValueTypes const value_types;
};


template<>
struct TypeTraits<int8_t>
{
    static ValueType const value_type;

    static ValueTypes const value_types;
};


template<>
struct TypeTraits<uint8_t>
{
    static ValueType const value_type;

    static ValueTypes const value_types;
};


template<>
struct TypeTraits<int16_t>
{
    static ValueType const value_type;

    static ValueTypes const value_types;
};


template<>
struct TypeTraits<uint16_t>
{
    static ValueType const value_type;

    static ValueTypes const value_types;
};


template<>
struct TypeTraits<int32_t>
{
    static ValueType const value_type;

    static ValueTypes const value_types;
};


template<>
struct TypeTraits<uint32_t>
{
    static ValueType const value_type;

    static ValueTypes const value_types;
};


template<>
struct TypeTraits<int64_t>
{
    static ValueType const value_type;

    static ValueTypes const value_types;
};


template<>
struct TypeTraits<uint64_t>
{
    static ValueType const value_type;

    static ValueTypes const value_types;
};


template<>
struct TypeTraits<float>
{
    static ValueType const value_type;

    static ValueTypes const value_types;
};


template<>
struct TypeTraits<double>
{
    static ValueType const value_type;

    static ValueTypes const value_types;
};


template<>
struct TypeTraits<String>
{
    static ValueType const value_type;

    static ValueTypes const value_types;
};

} // namespace geoneric
