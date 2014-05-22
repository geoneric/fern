#pragma once
#include "fern/core/assert.h"
#include "fern/core/type_traits.h"


namespace fern {
namespace absolute {
namespace detail {
namespace dispatch {

template<
    class Value,
    class R,
    class ValueNumberCategory>
struct within_range
{
};


template<
    class Value,
    class Result>
struct within_range<
    Value,
    Result,
    unsigned_integer_tag>
{
    inline static constexpr bool calculate(
        Value const& /* value */,
        Result const& /* result */)
    {
        static_assert(sizeof(Value) <= sizeof(Result), "");  // For now.
        return true;
    }
};


template<
    class Value,
    class Result>
struct within_range<
    Value,
    Result,
    signed_integer_tag>
{
    inline static constexpr bool calculate(
        Value const& value,
        Result const& result)
    {
        static_assert(sizeof(value) <= sizeof(result), "");  // For now.
        FERN_STATIC_ASSERT(std::is_same, Value, Result)
        return value != min<Value>();
    }
};


template<
    class Value,
    class Result>
struct within_range<
    Value,
    Result,
    integer_tag>
{
    inline static bool calculate(
        Value const& value,
        Result const& result)
    {
        return within_range<Value, Result,
            typename TypeTraits<Value>::number_category>::calculate(value,
                result);
    }
};


template<
    class Value,
    class Result>
struct within_range<
    Value,
    Result,
    floating_point_tag>
{
    inline static constexpr bool calculate(
        Value const& /* value */,
        Result const& /* result */)
    {
        FERN_STATIC_ASSERT(std::is_same, Value, Result)

        return true;
    }
};

} // namespace dispatch
} // namespace detail
} // namespace absolute
} // namespace fern
