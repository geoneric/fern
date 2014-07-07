#pragma once
#include "fern/core/assert.h"
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/spatial/focal/detail/slope.h"


namespace fern {
namespace slope {

template<
    class Value,
    class Result>
class OutOfRangePolicy
{

    FERN_STATIC_ASSERT(std::is_arithmetic, Result)
    FERN_STATIC_ASSERT(std::is_floating_point, Result)

public:

    inline bool within_range(
        Result const& result) const
    {
        return std::isfinite(result);
    }

};

} // namespace slope


namespace spatial {

template<
    template<class, class> class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Value,
    class Result
>
void slope(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    FERN_STATIC_ASSERT(std::is_floating_point, value_type<Value>)
    FERN_STATIC_ASSERT(std::is_same, value_type<Result>, value_type<Value>)

    slope::detail::slope<OutOfRangePolicy>(input_no_data_policy,
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
void slope(
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    OutputNoDataPolicy output_no_data_policy;
    slope<OutOfRangePolicy>(InputNoDataPolicy(), output_no_data_policy,
        execution_policy, value, result);
}


template<
    class ExecutionPolicy,
    class Value,
    class Result
>
void slope(
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    using InputNoDataPolicy = SkipNoData;
    using OutputNoDataPolicy = DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;
    slope<unary::DiscardRangeErrors>(InputNoDataPolicy(),
        output_no_data_policy, execution_policy, value, result);
}

} // namespace spatial
} // namespace fern
