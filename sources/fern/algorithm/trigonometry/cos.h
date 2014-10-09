#pragma once
#include "fern/core/assert.h"
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/trigonometry/detail/cos.h"


namespace fern {
namespace algorithm {
namespace cos {

/*!
    @ingroup    fern_algorithm_trigonometry_group
    @brief      Out-of-domain policy for fern::algorithm::trigonometry::cos
                algorithm.
    @sa         @ref fern_algorithm_policies_out_of_domain_policy

    Positive or negative infinity is considered out-of-domain for cos.

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
        return std::isfinite(value);
    }

};

} // namespace cos


namespace trigonometry {

//! Calculate the cosine of @a value and write the result to @a result.
/*!
    @ingroup    fern_algorithm_trigonometry_group
    @sa         fern::algorithm::cos::OutOfDomainPolicy,
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
void cos(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    FERN_STATIC_ASSERT(std::is_floating_point, value_type<Value>)
    FERN_STATIC_ASSERT(std::is_same, value_type<Result>, value_type<Value>)

    cos::detail::cos<OutOfDomainPolicy>(input_no_data_policy,
        output_no_data_policy, execution_policy, value, result);
}


/*!
    @ingroup    fern_algorithm_trigonometry_group
    @overload
*/
template<
    template<typename> class OutOfDomainPolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename ExecutionPolicy,
    typename Value,
    typename Result>
void cos(
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    OutputNoDataPolicy output_no_data_policy;
    cos<OutOfDomainPolicy>(InputNoDataPolicy(), output_no_data_policy,
        execution_policy, value, result);
}


/*!
    @ingroup    fern_algorithm_trigonometry_group
    @overload
*/
template<
    typename ExecutionPolicy,
    typename Value,
    typename Result>
void cos(
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    using InputNoDataPolicy = SkipNoData<>;
    using OutputNoDataPolicy = DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;
    cos<unary::DiscardDomainErrors>(InputNoDataPolicy(), output_no_data_policy,
        execution_policy, value, result);
}

} // namespace trigonometry
} // namespace algorithm
} // namespace fern
