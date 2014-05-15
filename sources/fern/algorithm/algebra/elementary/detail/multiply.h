#pragma once
#include <cmath>
#include "fern/core/assert.h"
#include "fern/core/type_traits.h"
#include "fern/algorithm/algebra/result_type.h"


namespace fern {
namespace multiply {
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
        Value1 const& value1,
        Value2 const& value2,
        Result const& /* result */)
    {
        FERN_STATIC_ASSERT(std::is_same,
            typename fern::Result<Value1, Value2>::type, Result)

        // TODO Optimization: depending on the types involved, overflow can
        //      never happen.

        // unsigned int * unsigned int
        return value2 == Value2(0) || value1 <= max<Result>() / value2;
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
    inline static /* constexpr */ bool calculate(
        Value1 const& value1,
        Value2 const& value2,
        Result const& /* result */)
    {
        FERN_STATIC_ASSERT(std::is_same,
            typename fern::Result<Value1, Value2>::type, Result)

        // TODO Optimization: depending on the types involved, overflow can
        //      never happen.

        // signed int * signed int
        if(value2 == Value2(0)) {
            return true;
        }
        else if(value1 > Value1(0)) {
            if(value2 > Value2(0)) {
                // value1 and value2 are positive.
                return value1 <= max<Result>() / value2;
            }
            else {
                // value1 is positive, value2 is negative.
                return -value1 >= max<Result>() / value2;
            }
        }
        else {
            if(value2 < Value2(0)) {
                // value1 and value2 are negative.
                return value1 >= max<Result>() / value2;
            }
            else {
                // value1 is zero or negative, value2 is positive.
                return value1 >= -max<Result>() / value2;
            }
        }
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
    inline static bool calculate(
        Value1 const& value1,
        Value2 const& value2,
        Result const& /* result */)
    {
        FERN_STATIC_ASSERT(std::is_same, typename fern::Result<Value1, Value2>
            ::type, Result)

        // TODO Optimization: depending on the types involved, overflow can
        //      never happen.

        // unsigned * signed
        if(value2 == Value2(0)) {
            return true;
        }
        else if(value2 > 0) {
            return value1 <= max<Result>() / value2;
        }
        else {
            return value1 <= -max<Result>() / value2;
        }
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
        // signed * unsigned
        return within_range<Value2, Value1, Result,
            typename TypeTraits<Value2>::number_category,
            typename TypeTraits<Value1>::number_category>::calculate(value2,
                value1, result);
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
    inline static bool calculate(
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

        // float * float
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
    inline static bool calculate(
        Value1 const& /* value1 */,
        Value2 const& /* value2 */,
        Result const& result)
    {
        FERN_STATIC_ASSERT(std::is_same,
            typename fern::Result<Value1, Value2>::type, Result)

        // integral * float
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
    inline static bool calculate(
        Value1 const& /* value1 */,
        Value2 const& /* values2 */,
        Result const& result)
    {
        FERN_STATIC_ASSERT(std::is_same,
            typename fern::Result<Value1, Value2>::type, Result)

        // float * integral
        return std::isfinite(result);
    }
};

} // namespace dispatch
} // namespace detail
} // namespace multiply
} // namespace fern
