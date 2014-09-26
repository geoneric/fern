#pragma once
#include "fern/core/assert.h"
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/trigonometry/detail/cos.h"


namespace fern {
namespace cos {

//! Out-of-domain policy for fern::trigonometry::cos algorithm.
/*!
    Positive or negative infinity is considered out-of-domain for cos.

    \a Value must be a floating point.

    \sa            @ref fern_algorithm_policies_out_of_domain_policy
*/
template<
    class Value>
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

//! Calculate the cosine of \a value and write the result to \a result.
/*!
    \ingroup       trigonometry
    \sa            fern::cos::OutOfDomainPolicy, fern::unary_local_operation,
                   @ref fern_algorithm_trigonometry

    The value types of \a value and \a result must be floating point and the
    same.
*/
template<
    template<class> class OutOfDomainPolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Value,
    class Result>
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
    \ingroup       trigonometry
    \overload
*/
template<
    template<class> class OutOfDomainPolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Value,
    class Result>
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
    \ingroup       trigonometry
    \overload
*/
template<
    class ExecutionPolicy,
    class Value,
    class Result>
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
} // namespace fern
