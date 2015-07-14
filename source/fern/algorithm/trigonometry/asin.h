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
#include "fern/algorithm/trigonometry/detail/asin.h"


namespace fern {
namespace algorithm {
namespace asin {

/*!
    @ingroup    fern_algorithm_trigonometry_group
    @brief      Out-of-domain policy for fern::algorithm::trigonometry::asin
                algorithm.
    @sa         @ref fern_algorithm_policies_out_of_domain_policy

    All values outside of the range [-1.0, 1.0] are considered out of the
    domain.

    @a Value must be a signed integral or a floating point type.
*/
template<
    typename Value>
class OutOfDomainPolicy
{

    static_assert(
        std::is_signed<Value>::value ||
        std::is_floating_point<Value>::value, "");

public:

    inline static bool within_domain(
        Value const& value)
    {
        return
            value >= static_cast<Value>(-1) &&
            value <= static_cast<Value>(1)
            ;
    }

};

} // namespace asin


namespace trigonometry {

//! Calculate the arc cosine of @a value and write the result to @a result.
/*!
    @ingroup    fern_algorithm_trigonometry_group
    @sa         fern::algorithm::asin::OutOfDomainPolicy,
                fern::algorithm::unary_local_operation

    The value types of @a value and @a result must be floating point and the
    same.
*/
template<
    template<typename> class OutOfDomainPolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename ExecutionPolicy,
    typename Value,
    typename Result>
void asin(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy& execution_policy,
    Value const& value,
    Result& result)
{
    FERN_STATIC_ASSERT(std::is_floating_point, value_type<Value>)
    FERN_STATIC_ASSERT(std::is_same, value_type<Result>, value_type<Value>)

    asin::detail::asin<OutOfDomainPolicy>(input_no_data_policy,
        output_no_data_policy, execution_policy, value, result);
}


/*!
    @ingroup    fern_algorithm_trigonometry_group
    @overload
*/
template<
    typename ExecutionPolicy,
    typename Value,
    typename Result>
void asin(
    ExecutionPolicy& execution_policy,
    Value const& value,
    Result& result)
{
    using InputNoDataPolicy = InputNoDataPolicies<SkipNoData>;
    using OutputNoDataPolicy = DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;
    asin<unary::DiscardDomainErrors>(InputNoDataPolicy{{}},
        output_no_data_policy, execution_policy, value, result);
}

} // namespace trigonometry
} // namespace algorithm
} // namespace fern
