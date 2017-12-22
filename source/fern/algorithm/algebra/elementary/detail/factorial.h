// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <cmath>
#include "fern/core/assert.h"
#include "fern/core/math.h"
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/core/unary_local_operation.h"


namespace fern {
namespace algorithm {
namespace factorial {
namespace detail {
namespace dispatch {

template<
    typename Value,
    typename ArgumentNumberCategory>
struct within_domain
{

    static bool    calculate           (Value const& value);

};


template<
    typename Value>
struct within_domain<
    Value,
    fern::integer_tag>
{

    inline static constexpr bool calculate(
        Value const& value)
    {
        return value >= Value{0};
    }

};


template<
    typename Value>
struct within_domain<
    Value,
    fern::floating_point_tag>
{

    inline static constexpr bool calculate(
        Value const& value)
    {
        return value >= Value{0.0} && is_equal(std::trunc(value), value);
    }

};


template<
    typename Value,
    typename Result,
    typename ArgumentNumberCategory>
struct within_range
{

    static bool    calculate           (Value const& value,
                                        Result const& result);

};


template<
    typename Value,
    typename Result>
struct within_range<
    Value,
    Result,
    fern::integer_tag>
{

    inline static constexpr bool calculate(
        Value const& value,
        Result const& result)
    {
        // Calculate the result as if the argument was a floating point and
        // compare the results. If the integral result does not equal the
        // floating point result, then the integral result has overflown.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
        return std::round(std::tgamma(value + 1)) ==
            static_cast<double>(result);
#pragma GCC diagnostic pop
    }

};


template<
    typename Value,
    typename Result>
struct within_range<
    Value,
    Result,
    fern::floating_point_tag>
{

    inline static constexpr bool calculate(
        Value const& /* value */,
        Result const& result)
    {
        return std::isfinite(result);
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
        FERN_STATIC_ASSERT(std::is_same, Result, Value)

        result = static_cast<Value>(std::round(std::tgamma(value + Value{1})));
    }

};


template<
    template<typename> class OutOfDomainPolicy,
    template<typename, typename> class OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename ExecutionPolicy,
    typename Value,
    typename Result
>
void factorial(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy& execution_policy,
    Value const& value,
    Result& result)
{
    unary_local_operation<Algorithm,
        OutOfDomainPolicy, OutOfRangePolicy>(
            input_no_data_policy, output_no_data_policy,
            execution_policy,
            value, result);
}

} // namespace detail
} // namespace factorial
} // namespace algorithm
} // namespace fern
