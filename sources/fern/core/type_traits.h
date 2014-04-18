#pragma once
#include "fern/core/value_types.h"


namespace fern {

// Type categories. Used in tag dispatching.
struct boolean_tag {};  // A boolean is not an integer in fern, it's a boolean.
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

    static bool const builtin = false;
};


template<>
struct TypeTraits<bool>
{
    using number_category = boolean_tag;

    static ValueType const value_type;

    static ValueTypes const value_types;

    static String const name;

    static bool const builtin = true;
};


template<>
struct TypeTraits<int8_t>
{
    using number_category = signed_integer_tag;

    static ValueType const value_type;

    static ValueTypes const value_types;

    static String const name;

    static bool const builtin = true;

    static int8_t const min;

    static int8_t const max;
};


template<>
struct TypeTraits<uint8_t>
{
    using number_category = unsigned_integer_tag;

    static ValueType const value_type;

    static ValueTypes const value_types;

    static String const name;

    static bool const builtin = true;

    static uint8_t const min;

    static uint8_t const max;
};


template<>
struct TypeTraits<int16_t>
{
    using number_category = signed_integer_tag;

    static ValueType const value_type;

    static ValueTypes const value_types;

    static String const name;

    static bool const builtin = true;

    static int16_t const min;

    static int16_t const max;
};


template<>
struct TypeTraits<uint16_t>
{
    using number_category = unsigned_integer_tag;

    static ValueType const value_type;

    static ValueTypes const value_types;

    static String const name;

    static bool const builtin = true;

    static uint16_t const min;

    static uint16_t const max;
};


template<>
struct TypeTraits<int32_t>
{
    using number_category = signed_integer_tag;

    static ValueType const value_type;

    static ValueTypes const value_types;

    static String const name;

    static bool const builtin = true;

    static int32_t const min;

    static int32_t const max;
};


template<>
struct TypeTraits<uint32_t>
{
    using number_category = unsigned_integer_tag;

    static ValueType const value_type;

    static ValueTypes const value_types;

    static String const name;

    static bool const builtin = true;

    static uint32_t const min;

    static uint32_t const max;
};


template<>
struct TypeTraits<int64_t>
{
    using number_category = signed_integer_tag;

    static ValueType const value_type;

    static ValueTypes const value_types;

    static String const name;

    static bool const builtin = true;

    static int64_t const min;

    static int64_t const max;
};


template<>
struct TypeTraits<uint64_t>
{
    using number_category = unsigned_integer_tag;

    static ValueType const value_type;

    static ValueTypes const value_types;

    static String const name;

    static bool const builtin = true;

    static uint64_t const min;

    static uint64_t const max;
};


template<>
struct TypeTraits<float>
{
    using number_category = floating_point_tag;

    static ValueType const value_type;

    static ValueTypes const value_types;

    static String const name;

    static bool const builtin = true;

    static float const min;

    static float const max;
};


template<>
struct TypeTraits<double>
{
    using number_category = floating_point_tag;

    static ValueType const value_type;

    static ValueTypes const value_types;

    static String const name;

    static bool const builtin = true;

    static double const min;

    static double const max;
};


template<>
struct TypeTraits<String>
{
    static ValueType const value_type;

    static ValueTypes const value_types;

    static String const name;

    static bool const builtin = true;
};


template<
    class T>
T min()
{
  return TypeTraits<T>::min;
}


template<
    class T>
T max()
{
  return TypeTraits<T>::max;
}

} // namespace fern
