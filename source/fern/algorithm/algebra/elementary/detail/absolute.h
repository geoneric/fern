// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
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
    typename Value,
    typename R,
    typename ValueNumberCategory>
struct within_range
{
};


template<
    typename Value,
    typename Result>
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
    typename Value,
    typename Result>
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
    typename Value,
    typename Result>
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
    typename Value,
    typename Result>
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
    typename Value,
    typename Enable=void>
struct AlgorithmByValueType
{

    /// static_assert(false, "Not implemented");

};


template<
    typename Value>
struct AlgorithmByValueType<
    Value,
    typename std::enable_if<std::is_unsigned<Value>::value>::type>
{

    FERN_STATIC_ASSERT(std::is_arithmetic, Value)
    FERN_STATIC_ASSERT(std::is_unsigned, Value)

    template<
        typename Result>
    inline static void apply(
        Value const& value,
        Result& result)
    {
        // Taking the absolute value of an unsigned integer is a no-op.
        result = static_cast<Result>(value);
    }

};


template<
    typename Value>
struct AlgorithmByValueType<
    Value,
    typename std::enable_if<!std::is_unsigned<Value>::value>::type>
{

    FERN_STATIC_ASSERT(std::is_arithmetic, Value)
    FERN_STATIC_ASSERT(!std::is_unsigned, Value)

    template<
        typename Result>
    inline static void apply(
        Value const& value,
        Result& result)
    {
        result = static_cast<Result>(std::abs(value));
    }

};

} // namespace dispatch


template<
    typename Value>
struct Algorithm
{

    template<
        typename Result>
    inline void operator()(
        Value const& value,
        Result& result) const
    {
        dispatch::AlgorithmByValueType<Value>::apply(value, result);
    }

};


template<
    template<typename, typename> class OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename ExecutionPolicy,
    typename Value,
    typename Result
>
void absolute(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy& execution_policy,
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
