#pragma once
#include "fern/core/assert.h"
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/trigonometry/detail/sin.h"


namespace fern {
namespace algorithm {
namespace sin {

/*!
    @ingroup    fern_algorithm_trigonometry_group
    @brief      Out-of-domain policy for fern::algorithm::trigonometry::sin
                algorithm.
    @sa         @ref fern_algorithm_policies_out_of_domain_policy

    Positive or negative infinity is considered out-of-domain for sin.

    - @a Value must be a floating point.
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

} // namespace sin


namespace trigonometry {

/*!
    @ingroup    fern_algorithm_trigonometry_group
    @brief      Calculate the sine of @a value and write the result to
                @a result.
    @sa         fern::algorithm::sin::OutOfDomainPolicy,
                fern::algorithm::unary_local_operation

    The value types of @a value and @a result must be floating point and the
    same.
*/
template<
    template<class> class OutOfDomainPolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Value,
    class Result>
void sin(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    FERN_STATIC_ASSERT(std::is_floating_point, value_type<Value>)
    FERN_STATIC_ASSERT(std::is_same, value_type<Result>, value_type<Value>)

    sin::detail::sin<OutOfDomainPolicy>(input_no_data_policy,
        output_no_data_policy, execution_policy, value, result);
}


/*!
    @ingroup    fern_algorithm_trigonometry_group
    @overload
*/
template<
    template<class> class OutOfDomainPolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Value,
    class Result>
void sin(
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    OutputNoDataPolicy output_no_data_policy;
    sin<OutOfDomainPolicy>(InputNoDataPolicy(), output_no_data_policy,
        execution_policy, value, result);
}


/*!
    @ingroup    fern_algorithm_trigonometry_group
    @overload
*/
template<
    class ExecutionPolicy,
    class Value,
    class Result>
void sin(
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    using InputNoDataPolicy = SkipNoData<>;
    using OutputNoDataPolicy = DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;
    sin<unary::DiscardDomainErrors>(InputNoDataPolicy(), output_no_data_policy,
        execution_policy, value, result);
}

} // namespace trigonometry
} // namespace algorithm
} // namespace fern
