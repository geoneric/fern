#pragma once
#include "fern/core/argument_traits.h"
#include "fern/core/assert.h"
#include "fern/core/base_class.h"
#include "fern/core/collection_traits.h"
#include "fern/core/thread_client.h"
#include "fern/algorithm/core/index_ranges.h"
#include "fern/algorithm/policy/execution_policy.h"


namespace fern {
namespace detail {

template<
    class OutOfDomainPolicy,
    class OutOfRangePolicy,
    class Algorithm,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value1,
    class Value2,
    class Result>
void operation_0d_0d(
    Algorithm const& algorithm,
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    Value1 const& value1,
    Value2 const& value2,
    Result& result)
{
    // Don't do anything if the input value is no-data. We assume
    // that input no-data values are already marked as such in the
    // result.
    if(!input_no_data_policy.is_no_data()) {
        const_reference<Value1> v1(fern::get(value1));
        const_reference<Value2> v2(fern::get(value2));

        if(!OutOfDomainPolicy::within_domain(v1, v2)) {
            // Input value is out of domain. Mark result value as
            // no-data. Don't change the result value.
            output_no_data_policy.mark_as_no_data();
        }
        else {
            reference<Result> r(fern::get(result));

            algorithm(v1, v2, r);

            if(!OutOfRangePolicy::within_range(v1, v2, r)) {
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
    class Value1,
    class Value2,
    class Result>
void operation_1d_0d(
    Algorithm const& algorithm,
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    IndexRanges<1> const& index_ranges,
    Value1 const& value1,
    Value2 const& value2,
    Result& result)
{
    for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end(); ++i) {

        // Don't do anything if the input value is no-data. We assume
        // that input no-data values are already marked as such in the
        // result.
        if(!input_no_data_policy.is_no_data(i)) {
            const_reference<Value1> v1(fern::get(value1, i));
            const_reference<Value2> v2(fern::get(value2));

            if(!OutOfDomainPolicy::within_domain(v1, v2)) {
                // Input value is out of domain. Mark result value as
                // no-data. Don't change the result value.
                output_no_data_policy.mark_as_no_data(i);
            }
            else {
                reference<Result> r(fern::get(result, i));

                algorithm(v1, v2, r);

                if(!OutOfRangePolicy::within_range(v1, v2, r)) {
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
    class Value1,
    class Value2,
    class Result>
void operation_0d_1d(
    Algorithm const& algorithm,
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    IndexRanges<1> const& index_ranges,
    Value1 const& value1,
    Value2 const& value2,
    Result& result)
{
    for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end(); ++i) {

        // Don't do anything if the input value is no-data. We assume
        // that input no-data values are already marked as such in the
        // result.
        if(!input_no_data_policy.is_no_data(i)) {
            const_reference<Value1> v1(fern::get(value1));
            const_reference<Value1> v2(fern::get(value2, i));

            if(!OutOfDomainPolicy::within_domain(v1, v2)) {
                // Input value is out of domain. Mark result value as
                // no-data. Don't change the result value.
                output_no_data_policy.mark_as_no_data(i);
            }
            else {
                reference<Result> r(fern::get(result, i));

                algorithm(v1, v2, r);

                if(!OutOfRangePolicy::within_range(v1, v2, r)) {
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
    class Value1,
    class Value2,
    class Result>
void operation_1d_1d(
    Algorithm const& algorithm,
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    IndexRanges<1> const& index_ranges,
    Value1 const& value1,
    Value2 const& value2,
    Result& result)
{
    for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end(); ++i) {

        // Don't do anything if the input value is no-data. We assume
        // that input no-data values are already marked as such in the
        // result.
        if(!input_no_data_policy.is_no_data(i)) {
            const_reference<Value1> v1(fern::get(value1, i));
            const_reference<Value1> v2(fern::get(value2, i));

            if(!OutOfDomainPolicy::within_domain(v1, v2)) {
                // Input value is out of domain. Mark result value as
                // no-data. Don't change the result value.
                output_no_data_policy.mark_as_no_data(i);
            }
            else {
                reference<Result> r(fern::get(result, i));

                algorithm(v1, v2, r);

                if(!OutOfRangePolicy::within_range(v1, v2, r)) {
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
    class Value1,
    class Value2,
    class Result>
void operation_2d_0d(
    Algorithm const& algorithm,
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    IndexRanges<2> const& index_ranges,
    Value1 const& value1,
    Value2 const& value2,
    Result& result)
{
    for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end(); ++i) {
        for(size_t j = index_ranges[1].begin(); j < index_ranges[1].end();
                ++j) {

            // Don't do anything if the input value is no-data. We assume
            // that input no-data values are already marked as such in the
            // result.
            if(!input_no_data_policy.is_no_data(i, j)) {
                const_reference<Value1> v1(fern::get(value1, i, j));
                const_reference<Value2> v2(fern::get(value2));

                if(!OutOfDomainPolicy::within_domain(v1, v2)) {
                    // Input value is out of domain. Mark result value as
                    // no-data. Don't change the result value.
                    output_no_data_policy.mark_as_no_data(i, j);
                }
                else {
                    reference<Result> r(fern::get(result, i, j));

                    algorithm(v1, v2, r);

                    if(!OutOfRangePolicy::within_range(v1, v2, r)) {
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


template<
    class OutOfDomainPolicy,
    class OutOfRangePolicy,
    class Algorithm,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value1,
    class Value2,
    class Result>
void operation_0d_2d(
    Algorithm const& algorithm,
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    IndexRanges<2> const& index_ranges,
    Value1 const& value1,
    Value2 const& value2,
    Result& result)
{
    for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end(); ++i) {
        for(size_t j = index_ranges[1].begin(); j < index_ranges[1].end();
                ++j) {

            // Don't do anything if the input value is no-data. We assume
            // that input no-data values are already marked as such in the
            // result.
            if(!input_no_data_policy.is_no_data(i, j)) {
                const_reference<Value1> v1(fern::get(value1));
                const_reference<Value2> v2(fern::get(value2, i, j));

                if(!OutOfDomainPolicy::within_domain(v1, v2)) {
                    // Input value is out of domain. Mark result value as
                    // no-data. Don't change the result value.
                    output_no_data_policy.mark_as_no_data(i, j);
                }
                else {
                    reference<Result> r(fern::get(result, i, j));

                    algorithm(v1, v2, r);

                    if(!OutOfRangePolicy::within_range(v1, v2, r)) {
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


template<
    class OutOfDomainPolicy,
    class OutOfRangePolicy,
    class Algorithm,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value1,
    class Value2,
    class Result>
void operation_2d_2d(
    Algorithm const& algorithm,
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    IndexRanges<2> const& index_ranges,
    Value1 const& value1,
    Value2 const& value2,
    Result& result)
{
    for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end(); ++i) {
        for(size_t j = index_ranges[1].begin(); j < index_ranges[1].end();
                ++j) {

            // Don't do anything if the input value is no-data. We assume
            // that input no-data values are already marked as such in the
            // result.
            if(!input_no_data_policy.is_no_data(i, j)) {
                const_reference<Value1> v1(fern::get(value1, i, j));
                const_reference<Value2> v2(fern::get(value2, i, j));

                if(!OutOfDomainPolicy::within_domain(v1, v2)) {
                    // Input value is out of domain. Mark result value as
                    // no-data. Don't change the result value.
                    output_no_data_policy.mark_as_no_data(i, j);
                }
                else {
                    reference<Result> r(fern::get(result, i, j));

                    algorithm(v1, v2, r);

                    if(!OutOfRangePolicy::within_range(v1, v2, r)) {
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
    class Value1,
    class Value2,
    class Result,
    class ExecutionPolicy,
    class Value1CollectionCategory,
    class Value2CollectionCategory>
class BinaryLocalOperation
{
};


template<
    class Algorithm,
    class OutOfDomainPolicy,
    class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value1,
    class Value2,
    class Result,
    class ExecutionPolicy>
struct BinaryLocalOperation<
    Algorithm,
    OutOfDomainPolicy,
    OutOfRangePolicy,
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value1,
    Value2,
    Result,
    ExecutionPolicy,
    array_0d_tag,
    array_0d_tag>

{

    // f(0d array, 0d array)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ExecutionPolicy const& /* execution_policy */,
        Value1 const& value1,
        Value2 const& value2,
        Result& result)
    {
        Algorithm algorithm;

        operation_0d_0d<OutOfDomainPolicy, OutOfRangePolicy>(algorithm,
            input_no_data_policy, output_no_data_policy, value1, value2,
            result);
    }

};


template<
    class Algorithm,
    class OutOfDomainPolicy,
    class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value1,
    class Value2,
    class Result>
struct BinaryLocalOperation<
    Algorithm,
    OutOfDomainPolicy,
    OutOfRangePolicy,
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value1,
    Value2,
    Result,
    SequentialExecutionPolicy,
    array_1d_tag,
    array_0d_tag>

{

    // f(1d array, 0d array)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SequentialExecutionPolicy const& /* execution_policy */,
        Value1 const& value1,
        Value2 const& value2,
        Result& result)
    {
        assert(fern::size(value1) == fern::size(result));

        Algorithm algorithm;

        operation_1d_0d<OutOfDomainPolicy, OutOfRangePolicy>(algorithm,
            input_no_data_policy, output_no_data_policy,
            IndexRanges<1>{IndexRange(0, fern::size(value1))}, value1, value2,
            result);
    }

};


template<
    class Algorithm,
    class OutOfDomainPolicy,
    class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value1,
    class Value2,
    class Result>
struct BinaryLocalOperation<
    Algorithm,
    OutOfDomainPolicy,
    OutOfRangePolicy,
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value1,
    Value2,
    Result,
    ParallelExecutionPolicy,
    array_1d_tag,
    array_0d_tag>

{

    // f(1d array, 0d array)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ParallelExecutionPolicy const& /* execution_policy */,
        Value1 const& value1,
        Value2 const& value2,
        Result& result)
    {
        assert(fern::size(value1) == fern::size(result));

        ThreadPool& pool(ThreadClient::pool());
        size_t const size = fern::size(value1);
        std::vector<IndexRanges<1>> ranges = index_ranges(pool.size(), size);
        std::vector<std::future<void>> futures;
        futures.reserve(ranges.size());

        Algorithm algorithm;

        for(auto const& block_range: ranges) {
            auto function = std::bind(
                operation_1d_0d<
                    OutOfDomainPolicy, OutOfRangePolicy,
                    Algorithm,
                    InputNoDataPolicy, OutputNoDataPolicy,
                    Value1, Value2, Result>,
                std::cref(algorithm),
                std::cref(input_no_data_policy),
                std::ref(output_no_data_policy), std::cref(block_range),
                std::cref(value1), std::cref(value2), std::ref(result));
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
    class Value1,
    class Value2,
    class Result>
struct BinaryLocalOperation<
    Algorithm,
    OutOfDomainPolicy,
    OutOfRangePolicy,
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value1,
    Value2,
    Result,
    SequentialExecutionPolicy,
    array_0d_tag,
    array_1d_tag>

{

    // f(0d array, 1d array)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SequentialExecutionPolicy const& /* execution_policy */,
        Value1 const& value1,
        Value2 const& value2,
        Result& result)
    {
        assert(fern::size(value2) == fern::size(result));

        Algorithm algorithm;

        operation_0d_1d<OutOfDomainPolicy, OutOfRangePolicy>(algorithm,
            input_no_data_policy, output_no_data_policy,
            IndexRanges<1>{IndexRange(0, fern::size(value2))}, value1, value2,
            result);
    }

};


template<
    class Algorithm,
    class OutOfDomainPolicy,
    class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value1,
    class Value2,
    class Result>
struct BinaryLocalOperation<
    Algorithm,
    OutOfDomainPolicy,
    OutOfRangePolicy,
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value1,
    Value2,
    Result,
    ParallelExecutionPolicy,
    array_0d_tag,
    array_1d_tag>

{

    // f(0d array, 1d array)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ParallelExecutionPolicy const& /* execution_policy */,
        Value1 const& value1,
        Value2 const& value2,
        Result& result)
    {
        ThreadPool& pool(ThreadClient::pool());
        size_t const size = fern::size(value2);
        std::vector<IndexRanges<1>> ranges = index_ranges(pool.size(), size);
        std::vector<std::future<void>> futures;
        futures.reserve(ranges.size());

        Algorithm algorithm;

        for(auto const& block_range: ranges) {
            auto function = std::bind(
                operation_0d_1d<
                    OutOfDomainPolicy, OutOfRangePolicy,
                    Algorithm,
                    InputNoDataPolicy, OutputNoDataPolicy,
                    Value1, Value2, Result>,
                std::cref(algorithm),
                std::cref(input_no_data_policy),
                std::ref(output_no_data_policy), std::cref(block_range),
                std::cref(value1), std::cref(value2), std::ref(result));
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
    class Value1,
    class Value2,
    class Result>
struct BinaryLocalOperation<
    Algorithm,
    OutOfDomainPolicy,
    OutOfRangePolicy,
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value1,
    Value2,
    Result,
    SequentialExecutionPolicy,
    array_1d_tag,
    array_1d_tag>

{

    // f(1d array, 1d array)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SequentialExecutionPolicy const& /* execution_policy */,
        Value1 const& value1,
        Value2 const& value2,
        Result& result)
    {
        assert(fern::size(value1) == fern::size(value2));
        assert(fern::size(value2) == fern::size(result));

        Algorithm algorithm;

        operation_1d_1d<OutOfDomainPolicy, OutOfRangePolicy>(algorithm,
            input_no_data_policy, output_no_data_policy,
            IndexRanges<1>{IndexRange(0, fern::size(value1))}, value1, value2,
            result);
    }

};


template<
    class Algorithm,
    class OutOfDomainPolicy,
    class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value1,
    class Value2,
    class Result>
struct BinaryLocalOperation<
    Algorithm,
    OutOfDomainPolicy,
    OutOfRangePolicy,
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value1,
    Value2,
    Result,
    ParallelExecutionPolicy,
    array_1d_tag,
    array_1d_tag>

{

    // f(1d array, 1d array)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ParallelExecutionPolicy const& /* execution_policy */,
        Value1 const& value1,
        Value2 const& value2,
        Result& result)
    {
        assert(fern::size(value1) == fern::size(value2));
        assert(fern::size(value2) == fern::size(result));

        ThreadPool& pool(ThreadClient::pool());
        size_t const size = fern::size(value1);
        std::vector<IndexRanges<1>> ranges = index_ranges(pool.size(), size);
        std::vector<std::future<void>> futures;
        futures.reserve(ranges.size());

        Algorithm algorithm;

        for(auto const& block_range: ranges) {
            auto function = std::bind(
                operation_1d_1d<
                    OutOfDomainPolicy, OutOfRangePolicy,
                    Algorithm,
                    InputNoDataPolicy, OutputNoDataPolicy,
                    Value1, Value2, Result>,
                std::cref(algorithm),
                std::cref(input_no_data_policy),
                std::ref(output_no_data_policy), std::cref(block_range),
                std::cref(value1), std::cref(value2), std::ref(result));
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
    class Value1,
    class Value2,
    class Result>
struct BinaryLocalOperation<
    Algorithm,
    OutOfDomainPolicy,
    OutOfRangePolicy,
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value1,
    Value2,
    Result,
    ExecutionPolicy,
    array_1d_tag,
    array_1d_tag>

{

    // f(1d array, 1d array)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ExecutionPolicy const& execution_policy,
        Value1 const& value1,
        Value2 const& value2,
        Result& result)
    {
        switch(execution_policy.which()) {
            case detail::sequential_execution_policy_id: {
                BinaryLocalOperation<
                    Algorithm,
                    OutOfDomainPolicy,
                    OutOfRangePolicy,
                    InputNoDataPolicy,
                    OutputNoDataPolicy,
                    Value1,
                    Value2,
                    Result,
                    SequentialExecutionPolicy,
                    array_1d_tag, array_1d_tag>::apply(
                        input_no_data_policy, output_no_data_policy,
                        detail::get_policy<SequentialExecutionPolicy>(
                            execution_policy),
                        value1, value2, result);
                break;
            }
            case detail::parallel_execution_policy_id: {
                BinaryLocalOperation<
                    Algorithm,
                    OutOfDomainPolicy,
                    OutOfRangePolicy,
                    InputNoDataPolicy,
                    OutputNoDataPolicy,
                    Value1,
                    Value2,
                    Result,
                    ParallelExecutionPolicy,
                    array_1d_tag, array_1d_tag>::apply(
                        input_no_data_policy, output_no_data_policy,
                        detail::get_policy<ParallelExecutionPolicy>(
                            execution_policy),
                        value1, value2, result);
                break;
            }
        }
    }

};


template<
    class Algorithm,
    class OutOfDomainPolicy,
    class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value1,
    class Value2,
    class Result>
struct BinaryLocalOperation<
    Algorithm,
    OutOfDomainPolicy,
    OutOfRangePolicy,
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value1,
    Value2,
    Result,
    SequentialExecutionPolicy,
    array_2d_tag,
    array_0d_tag>

{

    // f(2d array, 0d array)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SequentialExecutionPolicy const& /* execution_policy */,
        Value1 const& value1,
        Value2 const& value2,
        Result& result)
    {
        assert(fern::size(value1, 0) == fern::size(result, 0));
        assert(fern::size(value1, 1) == fern::size(result, 1));

        Algorithm algorithm;

        operation_2d_0d<OutOfDomainPolicy, OutOfRangePolicy>(algorithm,
            input_no_data_policy, output_no_data_policy,
            IndexRanges<2>{
                IndexRange(0, fern::size(value1, 0)),
                IndexRange(0, fern::size(value1, 1))
            }, value1, value2, result);
    }

};


template<
    class Algorithm,
    class OutOfDomainPolicy,
    class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value1,
    class Value2,
    class Result>
struct BinaryLocalOperation<
    Algorithm,
    OutOfDomainPolicy,
    OutOfRangePolicy,
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value1,
    Value2,
    Result,
    ParallelExecutionPolicy,
    array_2d_tag,
    array_0d_tag>

{

    // f(2d array, 0d array)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ParallelExecutionPolicy const& /* execution_policy */,
        Value1 const& value1,
        Value2 const& value2,
        Result& result)
    {
        assert(fern::size(value1, 0) == fern::size(result, 0));
        assert(fern::size(value1, 1) == fern::size(result, 1));

        ThreadPool& pool(ThreadClient::pool());
        size_t const size1 = fern::size(value1, 0);
        size_t const size2 = fern::size(value1, 1);
        std::vector<IndexRanges<2>> ranges = index_ranges(pool.size(),
            size1, size2);
        std::vector<std::future<void>> futures;
        futures.reserve(ranges.size());

        Algorithm algorithm;

        for(auto const& block_range: ranges) {
            auto function = std::bind(
                operation_2d_0d<
                    OutOfDomainPolicy, OutOfRangePolicy,
                    Algorithm,
                    InputNoDataPolicy, OutputNoDataPolicy,
                    Value1, Value2, Result>,
                std::cref(algorithm),
                std::cref(input_no_data_policy),
                std::ref(output_no_data_policy), std::cref(block_range),
                std::cref(value1), std::cref(value2), std::ref(result));
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
    class Value1,
    class Value2,
    class Result>
struct BinaryLocalOperation<
    Algorithm,
    OutOfDomainPolicy,
    OutOfRangePolicy,
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value1,
    Value2,
    Result,
    ExecutionPolicy,
    array_2d_tag,
    array_0d_tag>

{

    // f(2d array, 0d array)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ExecutionPolicy const& execution_policy,
        Value1 const& value1,
        Value2 const& value2,
        Result& result)
    {
        switch(execution_policy.which()) {
            case detail::sequential_execution_policy_id: {
                BinaryLocalOperation<
                    Algorithm,
                    OutOfDomainPolicy,
                    OutOfRangePolicy,
                    InputNoDataPolicy,
                    OutputNoDataPolicy,
                    Value1,
                    Value2,
                    Result,
                    SequentialExecutionPolicy,
                    array_2d_tag, array_0d_tag>::apply(
                        input_no_data_policy, output_no_data_policy,
                        detail::get_policy<SequentialExecutionPolicy>(
                            execution_policy),
                        value1, value2, result);
                break;
            }
            case detail::parallel_execution_policy_id: {
                BinaryLocalOperation<
                    Algorithm,
                    OutOfDomainPolicy,
                    OutOfRangePolicy,
                    InputNoDataPolicy,
                    OutputNoDataPolicy,
                    Value1,
                    Value2,
                    Result,
                    ParallelExecutionPolicy,
                    array_2d_tag, array_0d_tag>::apply(
                        input_no_data_policy, output_no_data_policy,
                        detail::get_policy<ParallelExecutionPolicy>(
                            execution_policy),
                        value1, value2, result);
                break;
            }
        }
    }

};


template<
    class Algorithm,
    class OutOfDomainPolicy,
    class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value1,
    class Value2,
    class Result>
struct BinaryLocalOperation<
    Algorithm,
    OutOfDomainPolicy,
    OutOfRangePolicy,
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value1,
    Value2,
    Result,
    SequentialExecutionPolicy,
    array_0d_tag,
    array_2d_tag>

{

    // f(0d array, 2d array)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SequentialExecutionPolicy const& /* execution_policy */,
        Value1 const& value1,
        Value2 const& value2,
        Result& result)
    {
        assert(fern::size(value2, 0) == fern::size(result, 0));
        assert(fern::size(value2, 1) == fern::size(result, 1));

        Algorithm algorithm;

        operation_0d_2d<OutOfDomainPolicy, OutOfRangePolicy>(algorithm,
            input_no_data_policy, output_no_data_policy,
            IndexRanges<2>{
                IndexRange(0, fern::size(value2, 0)),
                IndexRange(0, fern::size(value2, 1))
            }, value1, value2, result);
    }

};


template<
    class Algorithm,
    class OutOfDomainPolicy,
    class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value1,
    class Value2,
    class Result>
struct BinaryLocalOperation<
    Algorithm,
    OutOfDomainPolicy,
    OutOfRangePolicy,
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value1,
    Value2,
    Result,
    ParallelExecutionPolicy,
    array_0d_tag,
    array_2d_tag>

{

    // f(0d array, 2d array)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ParallelExecutionPolicy const& /* execution_policy */,
        Value1 const& value1,
        Value2 const& value2,
        Result& result)
    {
        assert(fern::size(value2, 0) == fern::size(result, 0));
        assert(fern::size(value2, 1) == fern::size(result, 1));

        ThreadPool& pool(ThreadClient::pool());
        size_t const size1 = fern::size(value2, 0);
        size_t const size2 = fern::size(value2, 1);
        std::vector<IndexRanges<2>> ranges = index_ranges(pool.size(),
            size1, size2);
        std::vector<std::future<void>> futures;
        futures.reserve(ranges.size());

        Algorithm algorithm;

        for(auto const& block_range: ranges) {
            auto function = std::bind(
                operation_0d_2d<
                    OutOfDomainPolicy, OutOfRangePolicy,
                    Algorithm,
                    InputNoDataPolicy, OutputNoDataPolicy,
                    Value1, Value2, Result>,
                std::cref(algorithm),
                std::cref(input_no_data_policy),
                std::ref(output_no_data_policy), std::cref(block_range),
                std::cref(value1), std::cref(value2), std::ref(result));
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
    class Value1,
    class Value2,
    class Result>
struct BinaryLocalOperation<
    Algorithm,
    OutOfDomainPolicy,
    OutOfRangePolicy,
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value1,
    Value2,
    Result,
    ExecutionPolicy,
    array_0d_tag,
    array_2d_tag>

{

    // f(0d array, 2d array)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ExecutionPolicy const& execution_policy,
        Value1 const& value1,
        Value2 const& value2,
        Result& result)
    {
        switch(execution_policy.which()) {
            case detail::sequential_execution_policy_id: {
                BinaryLocalOperation<
                    Algorithm,
                    OutOfDomainPolicy,
                    OutOfRangePolicy,
                    InputNoDataPolicy,
                    OutputNoDataPolicy,
                    Value1,
                    Value2,
                    Result,
                    SequentialExecutionPolicy,
                    array_0d_tag, array_2d_tag>::apply(
                        input_no_data_policy, output_no_data_policy,
                        detail::get_policy<SequentialExecutionPolicy>(
                            execution_policy),
                        value1, value2, result);
                break;
            }
            case detail::parallel_execution_policy_id: {
                BinaryLocalOperation<
                    Algorithm,
                    OutOfDomainPolicy,
                    OutOfRangePolicy,
                    InputNoDataPolicy,
                    OutputNoDataPolicy,
                    Value1,
                    Value2,
                    Result,
                    ParallelExecutionPolicy,
                    array_0d_tag, array_2d_tag>::apply(
                        input_no_data_policy, output_no_data_policy,
                        detail::get_policy<ParallelExecutionPolicy>(
                            execution_policy),
                        value1, value2, result);
                break;
            }
        }
    }

};


template<
    class Algorithm,
    class OutOfDomainPolicy,
    class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value1,
    class Value2,
    class Result>
struct BinaryLocalOperation<
    Algorithm,
    OutOfDomainPolicy,
    OutOfRangePolicy,
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value1,
    Value2,
    Result,
    SequentialExecutionPolicy,
    array_2d_tag,
    array_2d_tag>

{

    // f(2d array, 2d array)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SequentialExecutionPolicy const& /* execution_policy */,
        Value1 const& value1,
        Value2 const& value2,
        Result& result)
    {
        assert(fern::size(value1, 0) == fern::size(value2, 0));
        assert(fern::size(value1, 1) == fern::size(value2, 1));
        assert(fern::size(value1, 0) == fern::size(result, 0));
        assert(fern::size(value1, 1) == fern::size(result, 1));

        Algorithm algorithm;

        operation_2d_2d<OutOfDomainPolicy, OutOfRangePolicy>(algorithm,
            input_no_data_policy, output_no_data_policy,
            IndexRanges<2>{
                IndexRange(0, fern::size(value1, 0)),
                IndexRange(0, fern::size(value1, 1))
            }, value1, value2, result);
    }

};


template<
    class Algorithm,
    class OutOfDomainPolicy,
    class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value1,
    class Value2,
    class Result>
struct BinaryLocalOperation<
    Algorithm,
    OutOfDomainPolicy,
    OutOfRangePolicy,
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value1,
    Value2,
    Result,
    ParallelExecutionPolicy,
    array_2d_tag,
    array_2d_tag>

{

    // f(2d array, 2d array)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ParallelExecutionPolicy const& /* execution_policy */,
        Value1 const& value1,
        Value2 const& value2,
        Result& result)
    {
        assert(fern::size(value1, 0) == fern::size(value2, 0));
        assert(fern::size(value1, 1) == fern::size(value2, 1));
        assert(fern::size(value1, 0) == fern::size(result, 0));
        assert(fern::size(value1, 1) == fern::size(result, 1));

        ThreadPool& pool(ThreadClient::pool());
        size_t const size1 = fern::size(value1, 0);
        size_t const size2 = fern::size(value1, 1);
        std::vector<IndexRanges<2>> ranges = index_ranges(pool.size(),
            size1, size2);
        std::vector<std::future<void>> futures;
        futures.reserve(ranges.size());

        Algorithm algorithm;

        for(auto const& block_range: ranges) {
            auto function = std::bind(
                operation_2d_2d<
                    OutOfDomainPolicy, OutOfRangePolicy,
                    Algorithm,
                    InputNoDataPolicy, OutputNoDataPolicy,
                    Value1, Value2, Result>,
                std::cref(algorithm),
                std::cref(input_no_data_policy),
                std::ref(output_no_data_policy), std::cref(block_range),
                std::cref(value1), std::cref(value2), std::ref(result));
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
    class Value1,
    class Value2,
    class Result>
struct BinaryLocalOperation<
    Algorithm,
    OutOfDomainPolicy,
    OutOfRangePolicy,
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value1,
    Value2,
    Result,
    ExecutionPolicy,
    array_2d_tag,
    array_2d_tag>

{

    // f(2d array, 2d array)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ExecutionPolicy const& execution_policy,
        Value1 const& value1,
        Value2 const& value2,
        Result& result)
    {
        switch(execution_policy.which()) {
            case detail::sequential_execution_policy_id: {
                BinaryLocalOperation<
                    Algorithm,
                    OutOfDomainPolicy,
                    OutOfRangePolicy,
                    InputNoDataPolicy,
                    OutputNoDataPolicy,
                    Value1,
                    Value2,
                    Result,
                    SequentialExecutionPolicy,
                    array_2d_tag, array_2d_tag>::apply(
                        input_no_data_policy, output_no_data_policy,
                        detail::get_policy<SequentialExecutionPolicy>(
                            execution_policy),
                        value1, value2, result);
                break;
            }
            case detail::parallel_execution_policy_id: {
                BinaryLocalOperation<
                    Algorithm,
                    OutOfDomainPolicy,
                    OutOfRangePolicy,
                    InputNoDataPolicy,
                    OutputNoDataPolicy,
                    Value1,
                    Value2,
                    Result,
                    ParallelExecutionPolicy,
                    array_2d_tag, array_2d_tag>::apply(
                        input_no_data_policy, output_no_data_policy,
                        detail::get_policy<ParallelExecutionPolicy>(
                            execution_policy),
                        value1, value2, result);
                break;
            }
        }
    }

};

} // namespace dispatch
} // namespace detail


//! Function that executes a binary local operation.
/*!
    \tparam        Algorithm Class template of the operation to execute.
    \param[in]     value1 First input to pass to the operation.
    \param[in]     value2 Second input to pass to the operation.
    \param[out]    result Output that is written by the operation.
    \sa            fern::nullary_local_operation, fern::unary_local_operation,
                   fern::n_ary_local_operation

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
    class Value1,
    class Value2,
    class Result
>
void binary_local_operation(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy const& execution_policy,
    Value1 const& value1,
    Value2 const& value2,
    Result& result)
{
    detail::dispatch::BinaryLocalOperation<
        Algorithm<value_type<Value1>, value_type<Value2>>,
        OutOfDomainPolicy<value_type<Value1>, value_type<Value2>>,
        OutOfRangePolicy<value_type<Value1>, value_type<Value2>,
            value_type<Result>>,
        InputNoDataPolicy,
        OutputNoDataPolicy,
        Value1,
        Value2,
        Result,
        ExecutionPolicy,
        base_class<argument_category<Value1>, array_2d_tag>,
        base_class<argument_category<Value2>, array_2d_tag>>::apply(
            input_no_data_policy, output_no_data_policy, execution_policy,
            value1, value2, result);
}

} // namespace fern
