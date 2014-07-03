#pragma once
#include "fern/core/assert.h"
#include "fern/core/base_class.h"
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/core/detail/cast.h"


namespace fern {
namespace cast {

template<
    class Value,
    class Result>
class OutOfRangePolicy
{

    FERN_STATIC_ASSERT(std::is_arithmetic, Value)
    FERN_STATIC_ASSERT(std::is_arithmetic, Result)

public:

    inline static bool within_range(
        Value const& value,
        Result const& result)
    {
        using value_tag =  base_class<number_category<Value>, integer_tag>;
        using result_tag = base_class<number_category<Result>, integer_tag>;

        return detail::dispatch::within_range<Value, Result, value_tag,
            result_tag>::calculate(value, result);
    }

};


} // namespace cast


namespace core {

template<
    template<class, class> class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Value,
    class Result
>
void cast(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    FERN_STATIC_ASSERT(std::is_arithmetic, value_type<Value>)
    FERN_STATIC_ASSERT(std::is_arithmetic, value_type<Result>)

    cast::detail::cast<OutOfRangePolicy>(input_no_data_policy,
        output_no_data_policy, execution_policy, value, result);
}


template<
    template<class, class> class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Value,
    class Result
>
void cast(
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    OutputNoDataPolicy output_no_data_policy;
    cast<OutOfRangePolicy>(InputNoDataPolicy(), output_no_data_policy,
        execution_policy, value, result);
}


template<
    class ExecutionPolicy,
    class Value,
    class Result
>
void cast(
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    using InputNoDataPolicy = SkipNoData;
    using OutputNoDataPolicy = DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;
    cast<unary::DiscardRangeErrors>(InputNoDataPolicy(),
        output_no_data_policy, execution_policy, value, result);
}

} // namespace core
} // namespace fern
