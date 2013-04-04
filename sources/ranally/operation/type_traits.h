#pragma once
#include "ranally/operation/value_types.h"


namespace ranally {

template<
    class T>
struct TypeTraits
{
    // static ValueTypes const value_types;
};


template<>
struct TypeTraits<int8_t>
{
    static ValueTypes const value_type;
};

ValueTypes const TypeTraits<int8_t>::value_type(ValueTypes::INT8);


template<>
struct TypeTraits<uint8_t>
{
    static ValueTypes const value_type;
};

ValueTypes const TypeTraits<uint8_t>::value_type(ValueTypes::UINT8);


template<>
struct TypeTraits<int16_t>
{
    static ValueTypes const value_type;
};

ValueTypes const TypeTraits<int16_t>::value_type(ValueTypes::INT16);


template<>
struct TypeTraits<uint16_t>
{
    static ValueTypes const value_type;
};

ValueTypes const TypeTraits<uint16_t>::value_type(ValueTypes::UINT16);


template<>
struct TypeTraits<int32_t>
{
    static ValueTypes const value_type;
};

ValueTypes const TypeTraits<int32_t>::value_type(ValueTypes::INT32);


template<>
struct TypeTraits<uint32_t>
{
    static ValueTypes const value_type;
};

ValueTypes const TypeTraits<uint32_t>::value_type(ValueTypes::UINT32);


template<>
struct TypeTraits<int64_t>
{
    static ValueTypes const value_type;
};

ValueTypes const TypeTraits<int64_t>::value_type(ValueTypes::INT64);


template<>
struct TypeTraits<uint64_t>
{
    static ValueTypes const value_type;
};

ValueTypes const TypeTraits<uint64_t>::value_type(ValueTypes::UINT64);


template<>
struct TypeTraits<float>
{
    static ValueTypes const value_type;
};

ValueTypes const TypeTraits<float>::value_type(ValueTypes::FLOAT32);


template<>
struct TypeTraits<double>
{
    static ValueTypes const value_type;
};

ValueTypes const TypeTraits<double>::value_type(ValueTypes::FLOAT64);


template<>
struct TypeTraits<String>
{
    static ValueTypes const value_type;
};

ValueTypes const TypeTraits<String>::value_type(ValueTypes::STRING);

} // namespace ranally
