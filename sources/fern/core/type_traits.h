#pragma once
#include <limits>
#include "fern/core/value_types.h"


namespace fern {

// Numeric type categories. Used in tag dispatching.
struct integer_tag {};
struct signed_integer_tag: public integer_tag {};
struct unsigned_integer_tag: public integer_tag {};
struct floating_point_tag {};


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
    typedef signed_integer_tag number_category;

    static ValueType const value_type;

    static ValueTypes const value_types;

    static String const name;

    static int8_t const min = std::numeric_limits<int8_t>::min();

    static int8_t const max = std::numeric_limits<int8_t>::max();
};


template<>
struct TypeTraits<uint8_t>
{
    typedef unsigned_integer_tag number_category;

    static ValueType const value_type;

    static ValueTypes const value_types;

    static String const name;

    static uint8_t const min = std::numeric_limits<uint8_t>::min();

    static uint8_t const max = std::numeric_limits<uint8_t>::max();
};


template<>
struct TypeTraits<int16_t>
{
    typedef signed_integer_tag number_category;

    static ValueType const value_type;

    static ValueTypes const value_types;

    static String const name;

    static int16_t const min = std::numeric_limits<int16_t>::min();

    static int16_t const max = std::numeric_limits<int16_t>::max();
};


template<>
struct TypeTraits<uint16_t>
{
    typedef unsigned_integer_tag number_category;

    static ValueType const value_type;

    static ValueTypes const value_types;

    static String const name;

    // static uint16_t const min = std::numeric_limits<uint16_t>::min();

    // static uint16_t const max = std::numeric_limits<uint16_t>::max();

    static uint16_t const min;

    static uint16_t const max;
};


template<>
struct TypeTraits<int32_t>
{
    typedef signed_integer_tag number_category;

    static ValueType const value_type;

    static ValueTypes const value_types;

    static String const name;

    static int32_t const min = std::numeric_limits<int32_t>::min();

    static int32_t const max = std::numeric_limits<int32_t>::max();
};


template<>
struct TypeTraits<uint32_t>
{
    typedef unsigned_integer_tag number_category;

    static ValueType const value_type;

    static ValueTypes const value_types;

    static String const name;

    static uint32_t const min = std::numeric_limits<uint32_t>::min();

    static uint32_t const max = std::numeric_limits<uint32_t>::max();
};


template<>
struct TypeTraits<int64_t>
{
    typedef signed_integer_tag number_category;

    static ValueType const value_type;

    static ValueTypes const value_types;

    static String const name;

    static int64_t const min = std::numeric_limits<int64_t>::min();

    static int64_t const max = std::numeric_limits<int64_t>::max();
};


template<>
struct TypeTraits<uint64_t>
{
    typedef unsigned_integer_tag number_category;

    static ValueType const value_type;

    static ValueTypes const value_types;

    static String const name;

    static uint64_t const min = std::numeric_limits<uint64_t>::min();

    static uint64_t const max = std::numeric_limits<uint64_t>::max();
};


template<>
struct TypeTraits<float>
{
    typedef floating_point_tag number_category;

    static ValueType const value_type;

    static ValueTypes const value_types;

    static String const name;

    static float const min;

    static float const max;
};


template<>
struct TypeTraits<double>
{
    typedef floating_point_tag number_category;

    static ValueType const value_type;

    static ValueTypes const value_types;

    static String const name;

    static double const min;

    static double const max;
};


template<>
struct TypeTraits<String>
{
    static ValueType const value_type;

    static ValueTypes const value_types;

    static String const name;
};

} // namespace fern
