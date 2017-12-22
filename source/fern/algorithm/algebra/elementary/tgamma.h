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
#include "fern/core/math.h"
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/algebra/elementary/detail/tgamma.h"


namespace fern {
namespace algorithm {
namespace tgamma {

/*!
    @ingroup    fern_algorithm_algebra_elementary_group
    @brief      Out of domain policy for fern::algorithm::algebra::tgamma
                algorithm.
    @sa         @ref fern_algorithm_policies_out_of_domain_policy

    Non-positive integers are considered out-of-domain.

    The value types of @a value and @a result must be floating point and
    the same.
*/
template<
    typename Value>
class OutOfDomainPolicy
{

    FERN_STATIC_ASSERT(std::is_floating_point, Value)

public:

    inline static bool within_domain(
        Value const& value)
    {
        return value >= Value{0} || is_not_equal(std::trunc(value), value);
    }

};


/*!
    @ingroup    fern_algorithm_algebra_elementary_group
    @brief      Out of range policy for fern::algorithm::algebra::tgamma
                algorithm.
    @sa         fern::algorithm::DetectOutOfRangeByErrno,
                @ref fern_algorithm_policies_out_of_range_policy

    The value types of @a value and @a Result must be floating point and
    the same.
*/
template<
    typename Value,
    typename Result>
class OutOfRangePolicy
{

    FERN_STATIC_ASSERT(std::is_floating_point, value_type<Value>)
    FERN_STATIC_ASSERT(std::is_same, value_type<Value>, value_type<Result>)

public:

    inline static bool within_range(
        Value const& /* value */,
        Result const& result)
    {
        return std::isfinite(result);
    }

};

} // namespace tgamma


namespace algebra {

/*!
    @ingroup    fern_algorithm_algebra_elementary_group
    @brief      Calculate the gamma function of @a value and write the result
                to @a result.
    @sa         fern::algorithm::tgamma::OutOfRangePolicy,
                fern::algorithm::unary_local_operation,
                http://en.wikipedia.com/wiki/Gamma_function

    The value types of @a value and @a result must be floating point and the
    same.
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
void tgamma(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy& execution_policy,
    Value const& value,
    Result& result)
{
    FERN_STATIC_ASSERT(std::is_floating_point, value_type<Value>)
    FERN_STATIC_ASSERT(std::is_same, value_type<Result>, value_type<Value>)

    tgamma::detail::tgamma<OutOfDomainPolicy, OutOfRangePolicy>(
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
void tgamma(
    ExecutionPolicy& execution_policy,
    Value const& value,
    Result& result)
{
    using InputNoDataPolicy = InputNoDataPolicies<SkipNoData>;
    using OutputNoDataPolicy = DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;
    tgamma<unary::DiscardDomainErrors, unary::DiscardRangeErrors>(
        InputNoDataPolicy{{}}, output_no_data_policy, execution_policy,
        value, result);
}

} // namespace algebra
} // namespace algorithm
} // namespace fern
