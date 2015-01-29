#pragma once
#include "fern/core/assert.h"
#include "fern/core/type_traits.h"
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/trigonometry/detail/tan.h"


namespace fern {
namespace algorithm {
namespace tan {

/*!
    @ingroup    fern_algorithm_trigonometry_group
    @brief      Out-of-domain policy for fern::algorithm::trigonometry::tan
                algorithm.
    @sa         @ref fern_algorithm_policies_out_of_domain_policy

    Positive or negative infinity is considered out-of-domain for sin.

    Valid input values for tan are finite and not divisable by an odd number
    of times 0.5 * π.

    - @a Value must be a floating point.
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
        if(!std::isfinite(value)) {
            // All bets are off.
            return false;
        }
        else {
            Value remainder = std::remainder(value, fern::half_pi<Value>());

            if(remainder != Value(0)) {
                // Value is not divisable by a whole number of times 0.5 * pi.
                return true;
            }
            else {
                Value quotient = value / fern::half_pi<Value>();

                return int64_t(quotient) % 2 == 0;
            }
        }
    }

};

} // namespace tan


namespace trigonometry {

/*!
    @ingroup    fern_algorithm_trigonometry_group
    @brief      Calculate the tangent of @a value and write the result
                to @a result.
    @sa         fern::algorithm::tan::OutOfDomainPolicy,
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
void tan(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy& execution_policy,
    Value const& value,
    Result& result)
{
    FERN_STATIC_ASSERT(std::is_floating_point, value_type<Value>)
    FERN_STATIC_ASSERT(std::is_same, value_type<Result>, value_type<Value>)

    tan::detail::tan<OutOfDomainPolicy>(input_no_data_policy,
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
void tan(
    ExecutionPolicy& execution_policy,
    Value const& value,
    Result& result)
{
    using InputNoDataPolicy = InputNoDataPolicies<SkipNoData>;
    using OutputNoDataPolicy = DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;
    tan<unary::DiscardDomainErrors>(InputNoDataPolicy{{}},
        output_no_data_policy, execution_policy, value, result);
}

} // namespace trigonometry
} // namespace algorithm
} // namespace fern
