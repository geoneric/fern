#pragma once
#include "fern/core/assert.h"
#include "fern/core/type_traits.h"


namespace fern {
namespace cast {
namespace detail {
namespace dispatch {

template<
    class Value,
    class R,
    class ValueNumberCategory,
    class ResultNumberCategory>
struct within_range
{
};


template<
    class Value,
    class Result>
struct within_range<
    Value,
    Result,
    unsigned_integer_tag,
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
    signed_integer_tag,
    signed_integer_tag>
{
    inline static constexpr bool calculate(
        Value const& value,
        Result const& result)
    {
        static_assert(sizeof(value) <= sizeof(result), "");  // For now.

        return true;
    }
};


// TODO signed_integer -> unsigned_integer
// TODO unsigned_integer -> signed_integer


template<
    class Value,
    class Result>
struct within_range<
    Value,
    Result,
    integer_tag,
    integer_tag>
{
    inline static bool calculate(
        Value const& value,
        Result const& result)
    {
        return within_range<Value, Result,
            typename TypeTraits<Value>::number_category,
            typename TypeTraits<Result>::number_category>::calculate(
                value, result);
    }
};


template<
    class Value,
    class Result>
struct within_range<
    Value,
    Result,
    unsigned_integer_tag,
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


template<
    class Value,
    class Result>
struct within_range<
    Value,
    Result,
    signed_integer_tag,
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


template<
    class Value,
    class Result>
struct within_range<
    Value,
    Result,
    floating_point_tag,
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


// TODO floating_point -> signed_integer
// TODO floating_point -> unsigned_integer


} // namespace dispatch
} // namespace detail
} // namespace cast
} // namespace fern
