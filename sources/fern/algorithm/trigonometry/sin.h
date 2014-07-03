#pragma once
#include "fern/core/assert.h"
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/trigonometry/detail/sin.h"


namespace fern {
namespace sin {

template<
    class Value
>
class OutOfDomainPolicy
{

    FERN_STATIC_ASSERT(std::is_floating_point, Value)

public:

    inline bool within_domain(
        Value const& value) const
    {
        return std::isfinite(value);
    }

};

} // namespace sin


namespace trigonometry {

template<
    template<class> class OutOfDomainPolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Value,
    class Result
>
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


template<
    template<class> class OutOfDomainPolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Value,
    class Result
>
void sin(
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    OutputNoDataPolicy output_no_data_policy;
    sin<OutOfDomainPolicy>(InputNoDataPolicy(), output_no_data_policy,
        execution_policy, value, result);
}


template<
    class ExecutionPolicy,
    class Value,
    class Result
>
void sin(
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    using InputNoDataPolicy = SkipNoData;
    using OutputNoDataPolicy = DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;
    sin<unary::DiscardDomainErrors>(InputNoDataPolicy(), output_no_data_policy,
        execution_policy, value, result);
}

} // namespace trigonometry
} // namespace fern
