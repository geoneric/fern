#pragma once
#include "fern/core/assert.h"
#include "fern/core/type_traits.h"
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/core/binary_local_operation.h"
#include "fern/algorithm/core/result_type.h"


namespace fern {
namespace algorithm {
namespace add {
namespace detail {
namespace dispatch {

template<
    typename Value1,
    typename Value2,
    typename R,
    typename A1NumberCategory,
    typename A2NumberCategory>
struct within_range
{
};


template<
    typename Value1,
    typename Value2,
    typename Result>
struct within_range<
    Value1,
    Value2,
    Result,
    unsigned_integer_tag,
    unsigned_integer_tag>
{
    inline static constexpr bool calculate(
        Value1 const& value1,
        Value2 const& /* value2 */,
        Result const& result)
    {
        FERN_STATIC_ASSERT(std::is_same,
            typename fern::algorithm::Result<Value1, Value2>::type, Result)

        // unsigned + unsigned
        // Overflow if result is smaller than one of the operands.
        return !(result < value1);
    }
};


template<
    typename Value1,
    typename Value2,
    typename Result>
struct within_range<
    Value1,
    Value2,
    Result,
    signed_integer_tag,
    signed_integer_tag>
{
    inline static constexpr bool calculate(
        Value1 const& value1,
        Value2 const& value2,
        Result const& result)
    {
        FERN_STATIC_ASSERT(std::is_same,
            typename fern::algorithm::Result<Value1, Value2>::type, Result)

        // signed + signed
        // Overflow/underflow if sign of result is different.
        // This only works if overflow wraps, which is not guaranteed.
        // See http://www.airs.com/blog/archives/120 and gcc's
        // -fno-strict-overflow argument.
        return value2 > 0 ? !(result < value1) : !(result > value1);
    }
};


template<
    typename Value1,
    typename Value2,
    typename Result>
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
        Result const& result)
    {
        // unsigned + signed
        // Switch arguments and forward request.
        return within_range<Value2, Value1, Result, signed_integer_tag,
            unsigned_integer_tag>::calculate(value2, value1, result);
    }
};


template<
    typename Value1,
    typename Value2,
    typename Result>
struct within_range<
    Value1,
    Value2,
    Result,
    signed_integer_tag,
    unsigned_integer_tag>
{
    inline static constexpr bool calculate(
        Value1 const& value1,
        Value2 const& /* value2 */,
        Result const& result)
    {
        FERN_STATIC_ASSERT(std::is_same, typename fern::algorithm::Result<
            Value1, Value2>::type, Result)

        return value1 > 0 ? result >= value1 : result <= value1;
    }
};


template<
    typename Value1,
    typename Value2,
    typename Result>
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
    typename Value1,
    typename Value2,
    typename Result>
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
            typename fern::algorithm::Result<Value1, Value2>::type, Result)

        return std::isfinite(result);
    }
};


template<
    typename Value1,
    typename Value2,
    typename Result>
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
            typename fern::algorithm::Result<Value1, Value2>::type, Result)
        assert(std::isfinite(result));

        // integral + float
        return true;
    }
};


template<
    typename Value1,
    typename Value2,
    typename Result>
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
            typename fern::algorithm::Result<Value1, Value2>::type, Result)
        assert(std::isfinite(result));

        // float + integral
        return true;
    }
};

} // namespace dispatch


template<
    typename Value1,
    typename Value2>
struct Algorithm
{

    FERN_STATIC_ASSERT(std::is_arithmetic, Value1)
    FERN_STATIC_ASSERT(std::is_arithmetic, Value2)

    template<
        typename Result>
    inline void operator()(
        Value1 const& value1,
        Value2 const& value2,
        Result& result) const
    {
        result = static_cast<Result>(value1) + static_cast<Result>(value2);
    }

};


template<
    template<typename, typename, typename> class OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename ExecutionPolicy,
    typename Value1,
    typename Value2,
    typename Result
>
void add(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy const& execution_policy,
    Value1 const& value1,
    Value2 const& value2,
    Result& result)
{
    binary_local_operation<Algorithm,
        binary::DiscardDomainErrors, OutOfRangePolicy>(
            input_no_data_policy, output_no_data_policy,
            execution_policy,
            value1, value2, result);
}

} // namespace detail
} // namespace add
} // namespace algorithm
} // namespace fern
