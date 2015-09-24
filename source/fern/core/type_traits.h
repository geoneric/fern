// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <limits>
#include <string>
#include <boost/math/constants/constants.hpp>
#include "fern/core/value_types.h"


namespace fern {

// Type categories. Used in tag dispatching.
struct boolean_tag {};  // A boolean is not an integer in fern, it's a boolean.
struct char_tag {};  // A character is a ISO/ASCII character, not a number.
struct integer_tag {};
struct signed_integer_tag: public integer_tag {};
struct unsigned_integer_tag: public integer_tag {};
struct floating_point_tag {};


template<
    typename T>
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

    static std::string const name;

    static bool const builtin = true;
};


template<>
struct TypeTraits<char>
{
    using number_category = char_tag;

    static ValueType const value_type;

    static ValueTypes const value_types;

    static std::string const name;

    static bool const builtin = true;
};


template<>
struct TypeTraits<int8_t>
{
    using number_category = signed_integer_tag;

    static ValueType const value_type;

    static ValueTypes const value_types;

    static std::string const name;

    static bool const builtin = true;

    static constexpr int8_t min()
    {
        return std::numeric_limits<int8_t>::min();
    }

    static constexpr int8_t max()
    {
        return std::numeric_limits<int8_t>::max();
    }

    static constexpr int8_t no_data_value()
    {
        return min();
    }
};


template<>
struct TypeTraits<uint8_t>
{
    using number_category = unsigned_integer_tag;

    static ValueType const value_type;

    static ValueTypes const value_types;

    static std::string const name;

    static bool const builtin = true;

    static constexpr uint8_t min()
    {
        return std::numeric_limits<uint8_t>::min();
    }

    static constexpr uint8_t max()
    {
        return std::numeric_limits<uint8_t>::max();
    }

    static constexpr uint8_t no_data_value()
    {
        return max();
    }
};


template<>
struct TypeTraits<int16_t>
{
    using number_category = signed_integer_tag;

    static ValueType const value_type;

    static ValueTypes const value_types;

    static std::string const name;

    static bool const builtin = true;

    static constexpr int16_t min()
    {
        return std::numeric_limits<int16_t>::min();
    }

    static constexpr int16_t max()
    {
        return std::numeric_limits<int16_t>::max();
    }

    static constexpr int16_t no_data_value()
    {
        return min();
    }
};


template<>
struct TypeTraits<uint16_t>
{
    using number_category = unsigned_integer_tag;

    static ValueType const value_type;

    static ValueTypes const value_types;

    static std::string const name;

    static bool const builtin = true;

    static constexpr uint16_t min()
    {
        return std::numeric_limits<uint16_t>::min();
    }

    static constexpr uint16_t max()
    {
        return std::numeric_limits<uint16_t>::max();
    }

    static constexpr uint16_t no_data_value()
    {
        return max();
    }
};


template<>
struct TypeTraits<int32_t>
{
    using number_category = signed_integer_tag;

    static ValueType const value_type;

    static ValueTypes const value_types;

    static std::string const name;

    static bool const builtin = true;

    static constexpr int32_t min()
    {
        return std::numeric_limits<int32_t>::min();
    }

    static constexpr int32_t max()
    {
        return std::numeric_limits<int32_t>::max();
    }

    static constexpr int32_t no_data_value()
    {
        return min();
    }
};


template<>
struct TypeTraits<uint32_t>
{
    using number_category = unsigned_integer_tag;

    static ValueType const value_type;

    static ValueTypes const value_types;

    static std::string const name;

    static bool const builtin = true;

    static constexpr uint32_t min()
    {
        return std::numeric_limits<uint32_t>::min();
    }

    static constexpr uint32_t max()
    {
        return std::numeric_limits<uint32_t>::max();
    }

    static constexpr uint32_t no_data_value()
    {
        return max();
    }
};


template<>
struct TypeTraits<int64_t>
{
    using number_category = signed_integer_tag;

    static ValueType const value_type;

    static ValueTypes const value_types;

    static std::string const name;

    static bool const builtin = true;

    static constexpr int64_t min()
    {
        return std::numeric_limits<int64_t>::min();
    }

    static constexpr int64_t max()
    {
        return std::numeric_limits<int64_t>::max();
    }

    static constexpr int64_t no_data_value()
    {
        return min();
    }
};


template<>
struct TypeTraits<uint64_t>
{
    using number_category = unsigned_integer_tag;

    static ValueType const value_type;

    static ValueTypes const value_types;

    static std::string const name;

    static bool const builtin = true;

    static constexpr uint64_t min()
    {
        return std::numeric_limits<uint64_t>::min();
    }

    static constexpr uint64_t max()
    {
        return std::numeric_limits<uint64_t>::max();
    }

    static constexpr uint64_t no_data_value()
    {
        return max();
    }
};


template<>
struct TypeTraits<float>
{
    using number_category = floating_point_tag;

    static ValueType const value_type;

    static ValueTypes const value_types;

    static std::string const name;

    static bool const builtin = true;

    static constexpr float min()
    {
        return std::numeric_limits<float>::min();
    }

    static constexpr float max()
    {
        return std::numeric_limits<float>::max();
    }

    static_assert(std::numeric_limits<float>::has_quiet_NaN, "");

    static constexpr float nan()
    {
        return std::numeric_limits<float>::quiet_NaN();
    }

    static constexpr float infinity()
    {
        return std::numeric_limits<float>::infinity();
    }

    static constexpr float no_data_value()
    {
        return min();
    }
};


template<>
struct TypeTraits<double>
{
    using number_category = floating_point_tag;

    static ValueType const value_type;

    static ValueTypes const value_types;

    static std::string const name;

    static bool const builtin = true;

    static constexpr double min()
    {
        return std::numeric_limits<double>::min();
    }

    static constexpr double max()
    {
        return std::numeric_limits<double>::max();
    }

    static_assert(std::numeric_limits<double>::has_quiet_NaN, "");

    static constexpr double nan()
    {
        return std::numeric_limits<double>::quiet_NaN();
    }

    static constexpr double infinity()
    {
        return std::numeric_limits<double>::infinity();
    }


    static constexpr double no_data_value()
    {
        return min();
    }
};


template<>
struct TypeTraits<std::string>
{
    static ValueType const value_type;

    static ValueTypes const value_types;

    static std::string const name;

    static bool const builtin = true;
};


template<
    typename T>
constexpr inline T min()
{
    return TypeTraits<T>::min();
}


template<
    typename T>
constexpr inline T max()
{
    return TypeTraits<T>::max();
}


template<
    typename T>
constexpr inline T nan()
{
    return TypeTraits<T>::nan();
}


template<
    typename T>
constexpr inline T infinity()
{
    return TypeTraits<T>::infinity();
}


template<
    typename T>
constexpr inline T no_data_value()
{
    return TypeTraits<T>::no_data_value();
}


template<
    typename T>
inline bool is_no_data(
    T const& value)
{
    return value == no_data_value<T>();
}


template<
    typename T>
inline void set_no_data(
    T& value)
{
    value = no_data_value<T>();
}


template<
    typename T>
constexpr inline T pi()
{
  return boost::math::constants::pi<T>();
}


template<
    typename T>
constexpr inline T half_pi()
{
  return boost::math::constants::half_pi<T>();
}


template<
    typename T>
using number_category = typename TypeTraits<T>::number_category;


template<
    typename T>
constexpr inline ValueType value_type_id()
{
    return TypeTraits<T>::value_type;
}


template<class... T>
struct are_same;

template<class T>
struct are_same<T>: std::true_type { };

template<class T>
struct are_same<T, T>: std::true_type { };

template<class T, class U>
struct are_same<T, U>: std::false_type { };

template<class T, class U, class... V>
struct are_same<T, U, V...>: std::conditional<
    are_same<T, U>::value && are_same<U, V...>::value,
    std::true_type, std::false_type>::type { };

} // namespace fern
