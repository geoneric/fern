#pragma once
#include <cmath>  // abs(float)
#include <cstdlib>  // abs(int)
#include "fern/core/assert.h"
#include "fern/core/type_traits.h"
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/core/unary_local_operation.h"


namespace fern {
namespace algorithm {
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

        // On a 2's complement system the absolute value of the most negative
        // integral value is out of range.
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


template<
    class Value,
    class Enable=void>
struct AlgorithmByValueType
{

    /// static_assert(false, "Not implemented");

};


template<
    class Value>
struct AlgorithmByValueType<
    Value,
    typename std::enable_if<std::is_unsigned<Value>::value>::type>
{

    FERN_STATIC_ASSERT(std::is_arithmetic, Value)
    FERN_STATIC_ASSERT(std::is_unsigned, Value)

    template<
        class Result>
    inline static void apply(
        Value const& value,
        Result& result)
    {
        // Taking the absolute value of an unsigned integer is a no-op.
        result = static_cast<Result>(value);
    }

};


template<
    class Value>
struct AlgorithmByValueType<
    Value,
    typename std::enable_if<!std::is_unsigned<Value>::value>::type>
{

    FERN_STATIC_ASSERT(std::is_arithmetic, Value)
    FERN_STATIC_ASSERT(!std::is_unsigned, Value)

    template<
        class Result>
    inline static void apply(
        Value const& value,
        Result& result)
    {
        result = static_cast<Result>(std::abs(value));
    }

};

} // namespace dispatch


template<
    class Value>
struct Algorithm
{

    template<
        class Result>
    inline void operator()(
        Value const& value,
        Result& result) const
    {
        dispatch::AlgorithmByValueType<Value>::apply(value, result);
    }

};


template<
    template<class, class> class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Value,
    class Result
>
void absolute(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    unary_local_operation<Algorithm,
        unary::DiscardDomainErrors, OutOfRangePolicy>(
            input_no_data_policy, output_no_data_policy,
            execution_policy,
            value, result);
}

} // namespace detail
} // namespace absolute
} // namespace algorithm
} // namespace fern
