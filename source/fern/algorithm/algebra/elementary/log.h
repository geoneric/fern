// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/algorithm/algebra/elementary/detail/log.h"


namespace fern {
namespace algorithm {
namespace log {

/*!
    @ingroup    fern_algorithm_algebra_elementary_group
    @brief      Out of domain policy for fern::algorithm::algebra::log
                algorithm.
    @sa         @ref fern_algorithm_policies_out_of_domain_policy

    Values smaller than 0 are considered out-of-domain.
*/
template<
    typename Value>
class OutOfDomainPolicy
{

public:

    inline static bool within_domain(
        Value const& value)
    {
        return value >= Value(0);
    }

};

} // namespace log


namespace algebra {

/*!
    @ingroup    fern_algorithm_algebra_elementary_group
    @brief      For each value in @a value, compute the natural (base e)
                logarithm of that value, and write each result value to
                @a result.
    @sa         fern::algorithm::log::OutOfDomainPolicy,
                fern::algorithm::unary_local_operation,
                http://en.cppreference.com/w/cpp/numeric/math/log

    - The value types of @a value and @a result must be floating point and the
      same.
    - All input values not smaller than zero are considered within log's
      domain.
    - All result values are considered within the result's value type's range.
*/
template<
    template<typename> class OutOfDomainPolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename ExecutionPolicy,
    typename Value,
    typename Result
>
void log(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy& execution_policy,
    Value const& value,
    Result& result)
{
    FERN_STATIC_ASSERT(std::is_floating_point, value_type<Value>)
    FERN_STATIC_ASSERT(std::is_same, value_type<Result>, value_type<Value>)

    log::detail::log<OutOfDomainPolicy>(input_no_data_policy,
        output_no_data_policy, execution_policy, value, result);
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
void log(
    ExecutionPolicy& execution_policy,
    Value const& value,
    Result& result)
{
    using InputNoDataPolicy = InputNoDataPolicies<SkipNoData>;
    using OutputNoDataPolicy = DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;
    log<unary::DiscardDomainErrors>(InputNoDataPolicy{{}},
        output_no_data_policy, execution_policy, value, result);
}

} // namespace algebra
} // namespace algorithm
} // namespace fern
