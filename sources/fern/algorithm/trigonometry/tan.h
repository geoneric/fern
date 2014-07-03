#pragma once
#include "fern/core/assert.h"
#include "fern/core/type_traits.h"
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/trigonometry/detail/tan.h"


namespace fern {
namespace tan {

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
        // - value must be finite and
        // - value must not be divisable by an odd number of times 0.5 * pi.

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

template<
    template<class> class OutOfDomainPolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Value,
    class Result
>
void tan(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    FERN_STATIC_ASSERT(std::is_floating_point, value_type<Value>)
    FERN_STATIC_ASSERT(std::is_same, value_type<Result>, value_type<Value>)

    tan::detail::tan<OutOfDomainPolicy>(input_no_data_policy,
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
void tan(
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    OutputNoDataPolicy output_no_data_policy;
    tan<OutOfDomainPolicy>(InputNoDataPolicy(), output_no_data_policy,
        execution_policy, value, result);
}


template<
    class ExecutionPolicy,
    class Value,
    class Result
>
void tan(
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    using InputNoDataPolicy = SkipNoData;
    using OutputNoDataPolicy = DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;
    tan<unary::DiscardDomainErrors>(InputNoDataPolicy(), output_no_data_policy,
        execution_policy, value, result);
}

} // namespace trigonometry
} // namespace fern
