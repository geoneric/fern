#pragma once
#include <cmath>
#include "fern/core/assert.h"
#include "fern/core/type_traits.h"
#include "fern/algorithm/algebra/result_type.h"


namespace fern {
namespace divide {
namespace detail {
namespace dispatch {

template<
    class Value1,
    class Value2,
    class R,
    class A1NumberCategory,
    class A2NumberCategory>
struct within_range
{
};


template<
    class Value1,
    class Value2,
    class Result>
struct within_range<
    Value1,
    Value2,
    Result,
    unsigned_integer_tag,
    unsigned_integer_tag>
{
    inline static constexpr bool calculate(
        Value1 const& /* value1 */,
        Value2 const& /* value2 */,
        Result const& /* result */)
    {
        FERN_STATIC_ASSERT(std::is_same,
            typename fern::Result<Value1, Value2>::type, Result)

        // unsigned / signed
        return true;
    }
};


template<
    class Value1,
    class Value2,
    class Result>
struct within_range<
    Value1,
    Value2,
    Result,
    signed_integer_tag,
    signed_integer_tag>
{
    inline static constexpr bool calculate(
        Value1 const& /* value1 */,
        Value2 const& /* value2 */,
        Result const& /* result */)
    {
        FERN_STATIC_ASSERT(std::is_same,
            typename fern::Result<Value1, Value2>::type, Result)

        // signed / signed
        return true;
    }
};


template<
    class Value1,
    class Value2,
    class Result>
struct within_range<
    Value1,
    Value2,
    Result,
    unsigned_integer_tag,
    signed_integer_tag>
{
    inline static constexpr bool calculate(
        Value1 const& /* value1 */,
        Value2 const& /* value2 */,
        Result const& /* result */)
    {
        FERN_STATIC_ASSERT(std::is_same, typename fern::Result<Value1, Value2>
            ::type, Result)

        // unsigned / signed
        return true;
    }
};


template<
    class Value1,
    class Value2,
    class Result>
struct within_range<
    Value1,
    Value2,
    Result,
    signed_integer_tag,
    unsigned_integer_tag>
{
    inline static constexpr bool calculate(
        Value1 const& value1,
        Value2 const& value2,
        Result const& result)
    {
        FERN_STATIC_ASSERT(std::is_same, typename fern::Result<Value1, Value2>
            ::type, Result)

        // signed / unsigned
        return true;
    }
};


template<
    class Value1,
    class Value2,
    class Result>
struct within_range<
    Value1,
    Value2,
    Result,
    integer_tag,
    integer_tag>
{
    inline static constexpr bool calculate(
        Value1 const& value1,
        Value2 const& value2,
        Result const& result)
    {
        return within_range<Value1, Value2, Result,
            typename TypeTraits<Value1>::number_category,
            typename TypeTraits<Value2>::number_category>::calculate(value1,
                value2, result);
    }
};


template<
    class Value1,
    class Value2,
    class Result>
struct within_range<
    Value1,
    Value2,
    Result,
    floating_point_tag,
    floating_point_tag>
{
    inline static constexpr bool calculate(
        Value1 const& /* value1 */,
        Value2 const& /* value2 */,
        Result const& result)
    {
        FERN_STATIC_ASSERT(std::is_same,
            typename fern::Result<Value1, Value2>::type, Result)

        // float / float
        return std::isfinite(result);
    }
};


template<
    class Value1,
    class Value2,
    class Result>
struct within_range<
    Value1,
    Value2,
    Result,
    integer_tag,
    floating_point_tag>
{
    inline static constexpr bool calculate(
        Value1 const& /* value1 */,
        Value2 const& /* value2 */,
        Result const& result)
    {
        FERN_STATIC_ASSERT(std::is_same,
            typename fern::Result<Value1, Value2>::type, Result)

        // integral / float
        return std::isfinite(result);
    }
};


template<
    class Value1,
    class Value2,
    class Result>
struct within_range<
    Value1,
    Value2,
    Result,
    floating_point_tag,
    integer_tag>
{
    inline static constexpr bool calculate(
        Value1 const& /* value1 */,
        Value2 const& /* values2 */,
        Result const& result)
    {
        FERN_STATIC_ASSERT(std::is_same,
            typename fern::Result<Value1, Value2>::type, Result)

        // float / integral
        return std::isfinite(result);
    }
};

} // namespace dispatch
} // namespace detail
} // namespace divide
} // namespace fern
