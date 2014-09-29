#pragma once
#include "fern/core/argument_traits.h"
#include "fern/core/assert.h"
#include "fern/core/base_class.h"
#include "fern/core/collection_traits.h"
#include "fern/core/thread_client.h"
#include "fern/algorithm/core/index_ranges.h"
#include "fern/algorithm/policy/execution_policy.h"


namespace fern {
namespace algorithm {
namespace detail {

template<
    class OutOfDomainPolicy,
    class OutOfRangePolicy,
    class Algorithm,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value,
    class Result>
void operation_0d(
    Algorithm const& algorithm,
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    Value const& value,
    Result& result)
{
    // Don't do anything if the input value is no-data. We assume
    // that input no-data values are already marked as such in the
    // result.
    if(!input_no_data_policy.is_no_data()) {
        const_reference<Value> v(fern::get(value));

        if(!OutOfDomainPolicy::within_domain(v)) {
            // Input value is out of domain. Mark result value as
            // no-data. Don't change the result value.
            output_no_data_policy.mark_as_no_data();
        }
        else {
            reference<Result> r(fern::get(result));

            algorithm(v, r);

            if(!OutOfRangePolicy::within_range(v, r)) {
                // Result value is out-of-range. Mark result value as
                // no-data. Result value contains the out-of-range
                // value (this may be overridden by
                // output_no_data_policy, depending on its
                // implementation).
                output_no_data_policy.mark_as_no_data();
            }
        }
    }
}


template<
    class OutOfDomainPolicy,
    class OutOfRangePolicy,
    class Algorithm,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value,
    class Result>
void operation_1d(
    Algorithm const& algorithm,
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    IndexRanges<1> const& index_ranges,
    Value const& value,
    Result& result)
{
    for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end(); ++i) {

        // Don't do anything if the input value is no-data. We assume
        // that input no-data values are already marked as such in the
        // result.
        if(!input_no_data_policy.is_no_data(i)) {
            const_reference<Value> v(fern::get(value, i));

            if(!OutOfDomainPolicy::within_domain(v)) {
                // Input value is out of domain. Mark result value as
                // no-data. Don't change the result value.
                output_no_data_policy.mark_as_no_data(i);
            }
            else {
                reference<Result> r(fern::get(result, i));

                algorithm(v, r);

                if(!OutOfRangePolicy::within_range(v, r)) {
                    // Result value is out-of-range. Mark result value as
                    // no-data. Result value contains the out-of-range
                    // value (this may be overridden by
                    // output_no_data_policy, depending on its
                    // implementation).
                    output_no_data_policy.mark_as_no_data(i);
                }
            }
        }
    }
}


template<
    class OutOfDomainPolicy,
    class OutOfRangePolicy,
    class Algorithm,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value,
    class Result>
void operation_2d(
    Algorithm const& algorithm,
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    IndexRanges<2> const& index_ranges,
    Value const& value,
    Result& result)
{
    for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end(); ++i) {
        for(size_t j = index_ranges[1].begin(); j < index_ranges[1].end();
            ++j) {

            // Don't do anything if the input value is no-data. We assume
            // that input no-data values are already marked as such in the
            // result.
            if(!input_no_data_policy.is_no_data(i, j)) {
                const_reference<Value> v(fern::get(value, i, j));

                if(!OutOfDomainPolicy::within_domain(v)) {
                    // Input value is out of domain. Mark result value as
                    // no-data. Don't change the result value.
                    output_no_data_policy.mark_as_no_data(i, j);
                }
                else {
                    reference<Result> r(fern::get(result, i, j));

                    algorithm(v, r);

                    if(!OutOfRangePolicy::within_range(v, r)) {
                        // Result value is out-of-range. Mark result value as
                        // no-data. Result value contains the out-of-range
                        // value (this may be overridden by
                        // output_no_data_policy, depending on its
                        // implementation).
                        output_no_data_policy.mark_as_no_data(i, j);
                    }
                }
            }
        }
    }
}


namespace dispatch {

template<
    class Algorithm,
    class OutOfDomainPolicy,
    class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value,
    class Result,
    class ExecutionPolicy,
    class ValueCollectionCategory>
class UnaryLocalOperation
{
};


template<
    class Algorithm,
    class OutOfDomainPolicy,
    class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value,
    class Result,
    class ExecutionPolicy>
struct UnaryLocalOperation<
    Algorithm,
    OutOfDomainPolicy,
    OutOfRangePolicy,
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value,
    Result,
    ExecutionPolicy,
    array_0d_tag>

{

    // f(0d array)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ExecutionPolicy const& /* execution_policy */,
        Value const& value,
        Result& result)
    {
        Algorithm algorithm;

        operation_0d<OutOfDomainPolicy, OutOfRangePolicy>(algorithm,
            input_no_data_policy, output_no_data_policy, value, result);
    }

};


template<
    class Algorithm,
    class OutOfDomainPolicy,
    class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value,
    class Result>
struct UnaryLocalOperation<
    Algorithm,
    OutOfDomainPolicy,
    OutOfRangePolicy,
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value,
    Result,
    SequentialExecutionPolicy,
    array_1d_tag>

{

    // f(1d array)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SequentialExecutionPolicy const& /* execution_policy */,
        Value const& value,
        Result& result)
    {
        assert(fern::size(value) == fern::size(result));

        Algorithm algorithm;

        operation_1d<OutOfDomainPolicy, OutOfRangePolicy>(algorithm,
            input_no_data_policy, output_no_data_policy,
            IndexRanges<1>{IndexRange(0, fern::size(value))}, value, result);
    }

};


template<
    class Algorithm,
    class OutOfDomainPolicy,
    class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value,
    class Result>
struct UnaryLocalOperation<
    Algorithm,
    OutOfDomainPolicy,
    OutOfRangePolicy,
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value,
    Result,
    ParallelExecutionPolicy,
    array_1d_tag>

{

    // f(1d array)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ParallelExecutionPolicy const& /* execution_policy */,
        Value const& value,
        Result& result)
    {
        assert(fern::size(value) == fern::size(result));

        ThreadPool& pool(ThreadClient::pool());
        size_t const size = fern::size(value);
        std::vector<IndexRanges<1>> ranges = index_ranges(pool.size(), size);
        std::vector<std::future<void>> futures;
        futures.reserve(ranges.size());

        Algorithm algorithm;

        for(auto const& block_range: ranges) {
            auto function = std::bind(
                operation_1d<
                    OutOfDomainPolicy, OutOfRangePolicy,
                    Algorithm,
                    InputNoDataPolicy, OutputNoDataPolicy,
                    Value, Result>,
                std::cref(algorithm),
                std::cref(input_no_data_policy),
                std::ref(output_no_data_policy), std::cref(block_range),
                std::cref(value), std::ref(result));
            futures.emplace_back(pool.submit(function));
        }

        for(auto& future: futures) {
            future.get();
        }
    }

};


template<
    class Algorithm,
    class OutOfDomainPolicy,
    class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value,
    class Result>
struct UnaryLocalOperation<
    Algorithm,
    OutOfDomainPolicy,
    OutOfRangePolicy,
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value,
    Result,
    SequentialExecutionPolicy,
    array_2d_tag>

{

    // f(2d array)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SequentialExecutionPolicy const& /* execution_policy */,
        Value const& value,
        Result& result)
    {
        assert(fern::size(value, 0) == fern::size(result, 0));
        assert(fern::size(value, 1) == fern::size(result, 1));

        Algorithm algorithm;

        operation_2d<OutOfDomainPolicy, OutOfRangePolicy>(algorithm,
            input_no_data_policy, output_no_data_policy,
            IndexRanges<2>{
                IndexRange(0, fern::size(value, 0)),
                IndexRange(0, fern::size(value, 1))
            }, value, result);
    }

};


template<
    class Algorithm,
    class OutOfDomainPolicy,
    class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value,
    class Result>
struct UnaryLocalOperation<
    Algorithm,
    OutOfDomainPolicy,
    OutOfRangePolicy,
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value,
    Result,
    ParallelExecutionPolicy,
    array_2d_tag>

{

    // f(2d array)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ParallelExecutionPolicy const& /* execution_policy */,
        Value const& value,
        Result& result)
    {
        assert(fern::size(value, 0) == fern::size(result, 0));
        assert(fern::size(value, 1) == fern::size(result, 1));

        ThreadPool& pool(ThreadClient::pool());
        size_t const size1 = fern::size(value, 0);
        size_t const size2 = fern::size(value, 1);
        std::vector<IndexRanges<2>> ranges = index_ranges(pool.size(),
            size1, size2);
        std::vector<std::future<void>> futures;
        futures.reserve(ranges.size());

        Algorithm algorithm;

        for(auto const& block_range: ranges) {
            auto function = std::bind(
                operation_2d<
                    OutOfDomainPolicy, OutOfRangePolicy,
                    Algorithm,
                    InputNoDataPolicy, OutputNoDataPolicy,
                    Value, Result>,
                std::cref(algorithm),
                std::cref(input_no_data_policy),
                std::ref(output_no_data_policy), std::cref(block_range),
                std::cref(value), std::ref(result));
            futures.emplace_back(pool.submit(function));
        }

        for(auto& future: futures) {
            future.get();
        }
    }

};


template<
    class Algorithm,
    class OutOfDomainPolicy,
    class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value,
    class Result>
struct UnaryLocalOperation<
    Algorithm,
    OutOfDomainPolicy,
    OutOfRangePolicy,
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value,
    Result,
    ExecutionPolicy,
    array_2d_tag>

{

    // f(2d array)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ExecutionPolicy const& execution_policy,
        Value const& value,
        Result& result)
    {
        switch(execution_policy.which()) {
            case detail::sequential_execution_policy_id: {
                detail::dispatch::UnaryLocalOperation<
                    Algorithm,
                    OutOfDomainPolicy,
                    OutOfRangePolicy,
                    InputNoDataPolicy,
                    OutputNoDataPolicy,
                    Value,
                    Result,
                    SequentialExecutionPolicy,
                    array_2d_tag>::apply(
                        input_no_data_policy, output_no_data_policy,
                        detail::get_policy<SequentialExecutionPolicy>(
                            execution_policy),
                        value, result);
                break;
            }
            case detail::parallel_execution_policy_id: {
                detail::dispatch::UnaryLocalOperation<
                    Algorithm,
                    OutOfDomainPolicy,
                    OutOfRangePolicy,
                    InputNoDataPolicy,
                    OutputNoDataPolicy,
                    Value,
                    Result,
                    ParallelExecutionPolicy,
                    array_2d_tag>::apply(
                        input_no_data_policy, output_no_data_policy,
                        detail::get_policy<ParallelExecutionPolicy>(
                            execution_policy),
                        value, result);
                break;
            }
        }
    }

};

} // namespace dispatch
} // namespace detail


//! Function that executes a unary local operation.
/*!
    \tparam        Algorithm Class template of the operation to execute.
    \param[in]     value Input to pass to the operation.
    \param[out]    result Output that is written by the operation.
    \sa            fern::nullary_local_operation, fern::binary_local_operation,
                   fern::n_ary_local_operation

    This function supports handling 0d, 1d and 2d values.

    This function supports sequential and parallel execution of the operation.
*/
template<
    template<class> class Algorithm,
    template<class> class OutOfDomainPolicy,
    template<class, class> class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Value,
    class Result
>
void unary_local_operation(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    detail::dispatch::UnaryLocalOperation<
        Algorithm<value_type<Value>>,
        OutOfDomainPolicy<value_type<Value>>,
        OutOfRangePolicy<value_type<Value>, value_type<Result>>,
        InputNoDataPolicy,
        OutputNoDataPolicy,
        Value,
        Result,
        ExecutionPolicy,
        base_class<argument_category<Value>, array_2d_tag>>::apply(
            input_no_data_policy, output_no_data_policy, execution_policy,
            value, result);
}

} // namespace algorithm
} // namespace fern
