// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/core/assert.h"
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/algebra/elementary/detail/factorial.h"


namespace fern {
namespace algorithm {
namespace factorial {

/*!
    @ingroup    fern_algorithm_algebra_elementary_group
    @brief      Out of domain policy for fern::algorithm::algebra::factorial
                algorithm.
    @sa         @ref fern_algorithm_policies_out_of_domain_policy

    The factorial is defined for non-negative integer values. All other values
    are considered out-of-domain.

    The value types of @a value and @a result must be integral or floating
    point, and the same.
*/
template<
    typename Value>
class OutOfDomainPolicy
{

    static_assert(std::is_integral<Value>::value ||
        std::is_floating_point<Value>::value, "");

public:

    inline static bool within_domain(
        Value const& value)
    {
        using value_tag = base_class<number_category<Value>, integer_tag>;

        return detail::dispatch::within_domain<Value, value_tag>::calculate(
            value);
    }

};


/*!
    @ingroup    fern_algorithm_algebra_elementary_group
    @brief      Out of range policy for fern::algorithm::algebra::factorial
                algorithm.
    @sa         fern::algorithm::DetectOutOfRangeByErrno,
                @ref fern_algorithm_policies_out_of_range_policy

    The value types of @a value and @a result must be integral or floating
    point, and the same.
*/
template<
    typename Value,
    typename Result>
class OutOfRangePolicy
{

    static_assert(std::is_integral<Value>::value ||
        std::is_floating_point<Value>::value, "");
    FERN_STATIC_ASSERT(std::is_same, value_type<Value>, value_type<Result>)

public:

    inline static bool within_range(
        Value const& value,
        Result const& result)
    {
        using value_tag = base_class<number_category<Value>, integer_tag>;

        return detail::dispatch::within_range<Value, Result, value_tag>
            ::calculate(value, result);
    }

};

} // namespace factorial


namespace algebra {

/*!
    @ingroup    fern_algorithm_algebra_elementary_group
    @brief      Calculate the factorial of @a value and write the result
                to @a result.
    @sa         fern::algorithm::factorial::OutOfRangePolicy,
                fern::algorithm::unary_local_operation,
                http://en.wikipedia.com/wiki/Factorial

    The value types of @a value and @a result must be integral or floating
    point, and the same.
*/
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
    static_assert(std::is_integral<Value>::value ||
        std::is_floating_point<Value>::value, "");
    FERN_STATIC_ASSERT(std::is_same, value_type<Result>, value_type<Value>)

    factorial::detail::factorial<OutOfDomainPolicy, OutOfRangePolicy>(
        input_no_data_policy, output_no_data_policy, execution_policy,
        value, result);
}


/*!
    @ingroup    fern_algorithm_algebra_elementary_group
    @overload
*/
template<
    typename ExecutionPolicy,
    typename Value,
    typename Result
>
void factorial(
    ExecutionPolicy& execution_policy,
    Value const& value,
    Result& result)
{
    using InputNoDataPolicy = InputNoDataPolicies<SkipNoData>;
    using OutputNoDataPolicy = DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;
    factorial<unary::DiscardDomainErrors, unary::DiscardRangeErrors>(
        InputNoDataPolicy{{}}, output_no_data_policy, execution_policy,
        value, result);
}

} // namespace algebra
} // namespace algorithm
} // namespace fern
