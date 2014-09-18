#pragma once
#include "fern/algorithm/core/binary_local_operation.h"


namespace fern {
namespace detail {
namespace dispatch {

template<
    template<class, class> class Algorithm,
    template<class, class> class OutOfDomainPolicy,
    template<class, class, class> class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Result,
    class... Values>
struct NAryLocalOperation
{
};


template<
    template<class, class> class Algorithm,
    template<class, class> class OutOfDomainPolicy,
    template<class, class, class> class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Result,
    class Value1,
    class Value2>
struct NAryLocalOperation<
    Algorithm,
    OutOfDomainPolicy,
    OutOfRangePolicy,
    InputNoDataPolicy,
    OutputNoDataPolicy,
    ExecutionPolicy,
    Result,
    Value1,
    Value2>
{

    // Handle two argument values. Stop recursion.
    // f(r, v1, v2)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ExecutionPolicy const& execution_policy,
        Result& result,
        Value1 const& value1,
        Value2 const& value2)
    {
        binary_local_operation<Algorithm, OutOfDomainPolicy, OutOfRangePolicy>(
            input_no_data_policy, output_no_data_policy, execution_policy,
            value1, value2, result);
    }

};


template<
    template<class, class> class Algorithm,
    template<class, class> class OutOfDomainPolicy,
    template<class, class, class> class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Result,
    class Value1,
    class Value2,
    class Value3,
    class... Values>
struct NAryLocalOperation<
    Algorithm,
    OutOfDomainPolicy,
    OutOfRangePolicy,
    InputNoDataPolicy,
    OutputNoDataPolicy,
    ExecutionPolicy,
    Result,
    Value1,
    Value2,
    Value3,
    Values...>
{

    // f(r, v1, v2, v3)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ExecutionPolicy const& execution_policy,
        Result& result,
        Value1 const& value1,
        Value2 const& value2,
        Value2 const& value3,
        Values&&... values)
    {
        // Handle first two argument values. Store in result.
        // f(r, v1, v2);
        detail::dispatch::NAryLocalOperation<
            Algorithm,
            OutOfDomainPolicy,
            OutOfRangePolicy,
            InputNoDataPolicy,
            OutputNoDataPolicy,
            ExecutionPolicy,
            Result,
            Value1,
            Value2>::apply(
                input_no_data_policy, output_no_data_policy, execution_policy,
                result, value1, value2);

        // Process result value and third argument value. Store in result.
        // Recurse.
        // f(r, r, v3);
        detail::dispatch::NAryLocalOperation<
            Algorithm,
            OutOfDomainPolicy,
            OutOfRangePolicy,
            InputNoDataPolicy,
            OutputNoDataPolicy,
            ExecutionPolicy,
            Result,
            Result,
            Value3,
            Values...>::apply(
                input_no_data_policy, output_no_data_policy, execution_policy,
                result, result, value3, std::forward<Values>(values)...);
    }

};

} // namespace dispatch
} // namespace detail


//! Function that executes an n-ary local operation.
/*!
    \tparam        Algorithm Class template of the operation to execute.
    \param[out]    result Output that is written by the operation.
    \param[in]     values Argument values to pass to the operation.
    \sa            fern::nullary_local_operation, fern::unary_local_operation,
                   fern::binary_local_operation

    This function supports handling 0d, 1d and 2d values.

    This function supports sequential and parallel execution of the operation.
*/
template<
    template<class, class> class Algorithm,
    template<class, class> class OutOfDomainPolicy,
    template<class, class, class> class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Result,
    class... Values>
void n_ary_local_operation(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy const& execution_policy,
    Result& result,
    Values&&... values)
{
    detail::dispatch::NAryLocalOperation<
        Algorithm,
        OutOfDomainPolicy,
        OutOfRangePolicy,
        InputNoDataPolicy,
        OutputNoDataPolicy,
        ExecutionPolicy,
        Result,
        Values...>::apply(
            input_no_data_policy, output_no_data_policy, execution_policy,
            result, std::forward<Values>(values)...);
}

} // namespace fern
