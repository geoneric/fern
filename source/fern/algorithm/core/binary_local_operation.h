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
    typename OutOfDomainPolicy,
    typename OutOfRangePolicy,
    typename Algorithm,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value1,
    typename Value2,
    typename Result>
void operation_0d_0d(
    Algorithm const& algorithm,
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    Value1 const& value1,
    Value2 const& value2,
    Result& result)
{
    if(std::get<0>(input_no_data_policy).is_no_data() ||
            std::get<1>(input_no_data_policy).is_no_data()) {
        output_no_data_policy.mark_as_no_data();
    }
    else {
        const_reference<Value1> v1(get(value1));
        const_reference<Value2> v2(get(value2));

        if(!OutOfDomainPolicy::within_domain(v1, v2)) {
            // Input value is out of domain. Mark result value as
            // no-data. Don't change the result value.
            output_no_data_policy.mark_as_no_data();
        }
        else {
            reference<Result> r(get(result));

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
    typename OutOfDomainPolicy,
    typename OutOfRangePolicy,
    typename Algorithm,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value1,
    typename Value2,
    typename Result>
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

        if(std::get<0>(input_no_data_policy).is_no_data(i) ||
                std::get<1>(input_no_data_policy).is_no_data(i)) {
            output_no_data_policy.mark_as_no_data(i);
        }
        else {
            const_reference<Value1> v1(get(value1, i));
            const_reference<Value2> v2(get(value2));

            if(!OutOfDomainPolicy::within_domain(v1, v2)) {
                // Input value is out of domain. Mark result value as
                // no-data. Don't change the result value.
                output_no_data_policy.mark_as_no_data(i);
            }
            else {
                reference<Result> r(get(result, i));

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
    typename OutOfDomainPolicy,
    typename OutOfRangePolicy,
    typename Algorithm,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value1,
    typename Value2,
    typename Result>
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

        if(std::get<0>(input_no_data_policy).is_no_data(i) ||
                std::get<1>(input_no_data_policy).is_no_data(i)) {
            output_no_data_policy.mark_as_no_data(i);
        }
        else {
            const_reference<Value1> v1(get(value1));
            const_reference<Value1> v2(get(value2, i));

            if(!OutOfDomainPolicy::within_domain(v1, v2)) {
                // Input value is out of domain. Mark result value as
                // no-data. Don't change the result value.
                output_no_data_policy.mark_as_no_data(i);
            }
            else {
                reference<Result> r(get(result, i));

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
    typename OutOfDomainPolicy,
    typename OutOfRangePolicy,
    typename Algorithm,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value1,
    typename Value2,
    typename Result>
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

        if(std::get<0>(input_no_data_policy).is_no_data(i) ||
                std::get<1>(input_no_data_policy).is_no_data(i)) {
            output_no_data_policy.mark_as_no_data(i);
        }
        else {
            const_reference<Value1> v1(get(value1, i));
            const_reference<Value1> v2(get(value2, i));

            if(!OutOfDomainPolicy::within_domain(v1, v2)) {
                // Input value is out of domain. Mark result value as
                // no-data. Don't change the result value.
                output_no_data_policy.mark_as_no_data(i);
            }
            else {
                reference<Result> r(get(result, i));

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
    typename OutOfDomainPolicy,
    typename OutOfRangePolicy,
    typename Algorithm,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value1,
    typename Value2,
    typename Result>
void operation_2d_0d(
    Algorithm const& algorithm,
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    IndexRanges<2> const& index_ranges,
    Value1 const& value1,
    Value2 const& value2,
    Result& result)
{
    size_t index_;

    for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end(); ++i) {

        index_ = index(result, i, index_ranges[1].begin());

        for(size_t j = index_ranges[1].begin(); j < index_ranges[1].end();
                ++j) {

            if(std::get<0>(input_no_data_policy).is_no_data(index_) ||
                    std::get<1>(input_no_data_policy).is_no_data(index_)) {
                output_no_data_policy.mark_as_no_data(index_);
            }
            else {
                const_reference<Value1> v1(get(value1, index_));
                const_reference<Value2> v2(get(value2));

                if(!OutOfDomainPolicy::within_domain(v1, v2)) {
                    // Input value is out of domain. Mark result value as
                    // no-data. Don't change the result value.
                    output_no_data_policy.mark_as_no_data(index_);
                }
                else {
                    reference<Result> r(get(result, index_));

                    algorithm(v1, v2, r);

                    if(!OutOfRangePolicy::within_range(v1, v2, r)) {
                        // Result value is out-of-range. Mark result value as
                        // no-data. Result value contains the out-of-range
                        // value (this may be overridden by
                        // output_no_data_policy, depending on its
                        // implementation).
                        output_no_data_policy.mark_as_no_data(index_);
                    }
                }
            }

            ++index_;
        }
    }
}


template<
    typename OutOfDomainPolicy,
    typename OutOfRangePolicy,
    typename Algorithm,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value1,
    typename Value2,
    typename Result>
void operation_0d_2d(
    Algorithm const& algorithm,
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    IndexRanges<2> const& index_ranges,
    Value1 const& value1,
    Value2 const& value2,
    Result& result)
{
    size_t index_;

    for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end(); ++i) {

        index_ = index(result, i, index_ranges[1].begin());

        for(size_t j = index_ranges[1].begin(); j < index_ranges[1].end();
                ++j) {

            if(std::get<0>(input_no_data_policy).is_no_data(index_) ||
                    std::get<1>(input_no_data_policy).is_no_data(index_)) {
                output_no_data_policy.mark_as_no_data(index_);
            }
            else {
                const_reference<Value1> v1(get(value1));
                const_reference<Value2> v2(get(value2, index_));

                if(!OutOfDomainPolicy::within_domain(v1, v2)) {
                    // Input value is out of domain. Mark result value as
                    // no-data. Don't change the result value.
                    output_no_data_policy.mark_as_no_data(index_);
                }
                else {
                    reference<Result> r(get(result, index_));

                    algorithm(v1, v2, r);

                    if(!OutOfRangePolicy::within_range(v1, v2, r)) {
                        // Result value is out-of-range. Mark result value as
                        // no-data. Result value contains the out-of-range
                        // value (this may be overridden by
                        // output_no_data_policy, depending on its
                        // implementation).
                        output_no_data_policy.mark_as_no_data(index_);
                    }
                }
            }

            ++index_;
        }
    }
}


template<
    typename OutOfDomainPolicy,
    typename OutOfRangePolicy,
    typename Algorithm,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value1,
    typename Value2,
    typename Result>
void operation_2d_2d(
    Algorithm const& algorithm,
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    IndexRanges<2> const& index_ranges,
    Value1 const& value1,
    Value2 const& value2,
    Result& result)
{
    size_t index_;

    for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end(); ++i) {

        index_ = index(result, i, index_ranges[1].begin());

        for(size_t j = index_ranges[1].begin(); j < index_ranges[1].end();
                ++j) {

            if(std::get<0>(input_no_data_policy).is_no_data(index_) ||
                    std::get<1>(input_no_data_policy).is_no_data(index_)) {
                output_no_data_policy.mark_as_no_data(index_);
            }
            else {
                const_reference<Value1> v1(get(value1, index_));
                const_reference<Value2> v2(get(value2, index_));

                if(!OutOfDomainPolicy::within_domain(v1, v2)) {
                    // Input value is out of domain. Mark result value as
                    // no-data. Don't change the result value.
                    output_no_data_policy.mark_as_no_data(index_);
                }
                else {
                    reference<Result> r(get(result, index_));

                    algorithm(v1, v2, r);

                    if(!OutOfRangePolicy::within_range(v1, v2, r)) {
                        // Result value is out-of-range. Mark result value as
                        // no-data. Result value contains the out-of-range
                        // value (this may be overridden by
                        // output_no_data_policy, depending on its
                        // implementation).
                        output_no_data_policy.mark_as_no_data(index_);
                    }
                }
            }

            ++index_;
        }
    }
}


namespace dispatch {

template<
    typename Algorithm,
    typename OutOfDomainPolicy,
    typename OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value1,
    typename Value2,
    typename Result,
    typename ExecutionPolicy,
    typename Value1CollectionCategory,
    typename Value2CollectionCategory>
class BinaryLocalOperation
{
};


template<
    typename Algorithm,
    typename OutOfDomainPolicy,
    typename OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value1,
    typename Value2,
    typename Result,
    typename ExecutionPolicy>
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
    typename Algorithm,
    typename OutOfDomainPolicy,
    typename OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value1,
    typename Value2,
    typename Result>
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
        assert(size(value1) == size(result));

        Algorithm algorithm;

        operation_1d_0d<OutOfDomainPolicy, OutOfRangePolicy>(algorithm,
            input_no_data_policy, output_no_data_policy,
            IndexRanges<1>{IndexRange(0, size(value1))}, value1, value2,
            result);
    }

};


template<
    typename Algorithm,
    typename OutOfDomainPolicy,
    typename OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value1,
    typename Value2,
    typename Result>
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
        assert(size(value1) == size(result));

        ThreadPool& pool(ThreadClient::pool());
        size_t const size_ = size(value1);
        std::vector<IndexRanges<1>> ranges = index_ranges(pool.size(), size_);
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
    typename Algorithm,
    typename OutOfDomainPolicy,
    typename OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value1,
    typename Value2,
    typename Result>
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
        assert(size(value2) == size(result));

        Algorithm algorithm;

        operation_0d_1d<OutOfDomainPolicy, OutOfRangePolicy>(algorithm,
            input_no_data_policy, output_no_data_policy,
            IndexRanges<1>{IndexRange(0, size(value2))}, value1, value2,
            result);
    }

};


template<
    typename Algorithm,
    typename OutOfDomainPolicy,
    typename OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value1,
    typename Value2,
    typename Result>
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
        size_t const size_ = size(value2);
        std::vector<IndexRanges<1>> ranges = index_ranges(pool.size(), size_);
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
    typename Algorithm,
    typename OutOfDomainPolicy,
    typename OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value1,
    typename Value2,
    typename Result>
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
        assert(size(value1) == size(value2));
        assert(size(value2) == size(result));

        Algorithm algorithm;

        operation_1d_1d<OutOfDomainPolicy, OutOfRangePolicy>(algorithm,
            input_no_data_policy, output_no_data_policy,
            IndexRanges<1>{IndexRange(0, size(value1))}, value1, value2,
            result);
    }

};


template<
    typename Algorithm,
    typename OutOfDomainPolicy,
    typename OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value1,
    typename Value2,
    typename Result>
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
        assert(size(value1) == size(value2));
        assert(size(value2) == size(result));

        ThreadPool& pool(ThreadClient::pool());
        size_t const size_ = size(value1);
        std::vector<IndexRanges<1>> ranges = index_ranges(pool.size(), size_);
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
    typename Algorithm,
    typename OutOfDomainPolicy,
    typename OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value1,
    typename Value2,
    typename Result>
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
    typename Algorithm,
    typename OutOfDomainPolicy,
    typename OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value1,
    typename Value2,
    typename Result>
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
        assert(size(value1, 0) == size(result, 0));
        assert(size(value1, 1) == size(result, 1));

        Algorithm algorithm;

        operation_2d_0d<OutOfDomainPolicy, OutOfRangePolicy>(algorithm,
            input_no_data_policy, output_no_data_policy,
            IndexRanges<2>{
                IndexRange(0, size(value1, 0)),
                IndexRange(0, size(value1, 1))
            }, value1, value2, result);
    }

};


template<
    typename Algorithm,
    typename OutOfDomainPolicy,
    typename OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value1,
    typename Value2,
    typename Result>
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
        assert(size(value1, 0) == size(result, 0));
        assert(size(value1, 1) == size(result, 1));

        ThreadPool& pool(ThreadClient::pool());
        size_t const size1 = size(value1, 0);
        size_t const size2 = size(value1, 1);
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
    typename Algorithm,
    typename OutOfDomainPolicy,
    typename OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value1,
    typename Value2,
    typename Result>
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
    typename Algorithm,
    typename OutOfDomainPolicy,
    typename OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value1,
    typename Value2,
    typename Result>
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
        assert(size(value2, 0) == size(result, 0));
        assert(size(value2, 1) == size(result, 1));

        Algorithm algorithm;

        operation_0d_2d<OutOfDomainPolicy, OutOfRangePolicy>(algorithm,
            input_no_data_policy, output_no_data_policy,
            IndexRanges<2>{
                IndexRange(0, size(value2, 0)),
                IndexRange(0, size(value2, 1))
            }, value1, value2, result);
    }

};


template<
    typename Algorithm,
    typename OutOfDomainPolicy,
    typename OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value1,
    typename Value2,
    typename Result>
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
        assert(size(value2, 0) == size(result, 0));
        assert(size(value2, 1) == size(result, 1));

        ThreadPool& pool(ThreadClient::pool());
        size_t const size1 = size(value2, 0);
        size_t const size2 = size(value2, 1);
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
    typename Algorithm,
    typename OutOfDomainPolicy,
    typename OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value1,
    typename Value2,
    typename Result>
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
    typename Algorithm,
    typename OutOfDomainPolicy,
    typename OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value1,
    typename Value2,
    typename Result>
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
        assert(size(value1, 0) == size(value2, 0));
        assert(size(value1, 1) == size(value2, 1));
        assert(size(value1, 0) == size(result, 0));
        assert(size(value1, 1) == size(result, 1));

        Algorithm algorithm;

        operation_2d_2d<OutOfDomainPolicy, OutOfRangePolicy>(algorithm,
            input_no_data_policy, output_no_data_policy,
            IndexRanges<2>{
                IndexRange(0, size(value1, 0)),
                IndexRange(0, size(value1, 1))
            }, value1, value2, result);
    }

};


template<
    typename Algorithm,
    typename OutOfDomainPolicy,
    typename OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value1,
    typename Value2,
    typename Result>
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
        assert(size(value1, 0) == size(value2, 0));
        assert(size(value1, 1) == size(value2, 1));
        assert(size(value1, 0) == size(result, 0));
        assert(size(value1, 1) == size(result, 1));

        ThreadPool& pool(ThreadClient::pool());
        size_t const size1 = size(value1, 0);
        size_t const size2 = size(value1, 1);
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
    typename Algorithm,
    typename OutOfDomainPolicy,
    typename OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value1,
    typename Value2,
    typename Result>
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


/*!
    @ingroup    fern_algorithm_core_group
    @brief      Function that executes a binary local operation.
    @tparam     Algorithm Class template of the operation to execute.
    @param[in]  value1 First input to pass to the operation.
    @param[in]  value2 Second input to pass to the operation.
    @param[out] result Output that is written by the operation.
    @sa         fern::algorithm::nullary_local_operation,
                fern::algorithm::unary_local_operation,
                fern::algorithm::n_ary_local_operation

    This function supports handling 0d, 1d and 2d values.

    This function supports sequential and parallel execution of the operation.
*/
template<
    template<typename, typename> class Algorithm,
    template<typename, typename> class OutOfDomainPolicy,
    template<typename, typename, typename> class OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename ExecutionPolicy,
    typename Value1,
    typename Value2,
    typename Result
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

} // namespace algorithm
} // namespace fern
