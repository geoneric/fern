#pragma once
#include "fern/core/argument_traits.h"
#include "fern/core/assert.h"
#include "fern/core/base_class.h"
#include "fern/core/collection_traits.h"
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
    typename Value,
    typename Result>
void operation_0d(
    Algorithm const& algorithm,
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    Value const& value,
    Result& result)
{
    if(std::get<0>(input_no_data_policy).is_no_data()) {
        output_no_data_policy.mark_as_no_data();
    }
    else {
        const_reference<Value> v(get(value));

        if(!OutOfDomainPolicy::within_domain(v)) {
            // Input value is out of domain. Mark result value as
            // no-data. Don't change the result value.
            output_no_data_policy.mark_as_no_data();
        }
        else {
            reference<Result> r(get(result));

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
    typename OutOfDomainPolicy,
    typename OutOfRangePolicy,
    typename Algorithm,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result>
void operation_1d(
    Algorithm const& algorithm,
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    IndexRanges<1> const& index_ranges,
    Value const& value,
    Result& result)
{
    for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end(); ++i) {

        if(std::get<0>(input_no_data_policy).is_no_data(i)) {
            output_no_data_policy.mark_as_no_data(i);
        }
        else {
            const_reference<Value> v(get(value, i));

            if(!OutOfDomainPolicy::within_domain(v)) {
                // Input value is out of domain. Mark result value as
                // no-data. Don't change the result value.
                output_no_data_policy.mark_as_no_data(i);
            }
            else {
                reference<Result> r(get(result, i));

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
    typename OutOfDomainPolicy,
    typename OutOfRangePolicy,
    typename Algorithm,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result>
void operation_2d(
    Algorithm const& algorithm,
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    IndexRanges<2> const& index_ranges,
    Value const& value,
    Result& result)
{
    size_t index_;

    for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end(); ++i) {

        index_ = index(result, i, index_ranges[1].begin());

        for(size_t j = index_ranges[1].begin(); j < index_ranges[1].end();
            ++j) {

            if(std::get<0>(input_no_data_policy).is_no_data(index_)) {
                output_no_data_policy.mark_as_no_data(index_);
            }
            else {
                const_reference<Value> v(get(value, index_));

                if(!OutOfDomainPolicy::within_domain(v)) {
                    // Input value is out of domain. Mark result value as
                    // no-data. Don't change the result value.
                    output_no_data_policy.mark_as_no_data(index_);
                }
                else {
                    reference<Result> r(get(result, index_));

                    algorithm(v, r);

                    if(!OutOfRangePolicy::within_range(v, r)) {
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
    typename Value,
    typename Result,
    typename ExecutionPolicy,
    typename ValueCollectionCategory>
class UnaryLocalOperation
{
};


template<
    typename Algorithm,
    typename OutOfDomainPolicy,
    typename OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result,
    typename ExecutionPolicy>
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
        ExecutionPolicy& /* execution_policy */,
        Value const& value,
        Result& result)
    {
        Algorithm algorithm;

        operation_0d<OutOfDomainPolicy, OutOfRangePolicy>(algorithm,
            input_no_data_policy, output_no_data_policy, value, result);
    }

};


template<
    typename Algorithm,
    typename OutOfDomainPolicy,
    typename OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result>
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
        SequentialExecutionPolicy& /* execution_policy */,
        Value const& value,
        Result& result)
    {
        assert(size(value) == size(result));

        Algorithm algorithm;

        operation_1d<OutOfDomainPolicy, OutOfRangePolicy>(algorithm,
            input_no_data_policy, output_no_data_policy,
            IndexRanges<1>{IndexRange(0, size(value))}, value, result);
    }

};


template<
    typename Algorithm,
    typename OutOfDomainPolicy,
    typename OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result>
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
        ParallelExecutionPolicy& execution_policy,
        Value const& value,
        Result& result)
    {
        assert(size(value) == size(result));

        ThreadPool& pool(execution_policy.thread_pool());
        size_t const size_ = size(value);
        std::vector<IndexRanges<1>> ranges = index_ranges(pool.size(), size_);
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
    typename Algorithm,
    typename OutOfDomainPolicy,
    typename OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result>
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
        SequentialExecutionPolicy& /* execution_policy */,
        Value const& value,
        Result& result)
    {
        assert(size(value, 0) == size(result, 0));
        assert(size(value, 1) == size(result, 1));

        Algorithm algorithm;

        operation_2d<OutOfDomainPolicy, OutOfRangePolicy>(algorithm,
            input_no_data_policy, output_no_data_policy,
            IndexRanges<2>{
                IndexRange(0, size(value, 0)),
                IndexRange(0, size(value, 1))
            }, value, result);
    }

};


template<
    typename Algorithm,
    typename OutOfDomainPolicy,
    typename OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result>
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
        ParallelExecutionPolicy& execution_policy,
        Value const& value,
        Result& result)
    {
        assert(size(value, 0) == size(result, 0));
        assert(size(value, 1) == size(result, 1));

        ThreadPool& pool(execution_policy.thread_pool());
        size_t const size1 = size(value, 0);
        size_t const size2 = size(value, 1);
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
    typename Algorithm,
    typename OutOfDomainPolicy,
    typename OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result>
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
        ExecutionPolicy& execution_policy,
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
                        boost::get<SequentialExecutionPolicy>(execution_policy),
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
                        boost::get<ParallelExecutionPolicy>(execution_policy),
                        value, result);
                break;
            }
        }
    }

};

} // namespace dispatch
} // namespace detail


/*!
    @ingroup    fern_algorithm_core_group
    @brief      Function that executes a unary local operation.
    @tparam     Algorithm Class template of the operation to execute.
    @param[in]  value Input to pass to the operation.
    @param[out] result Output that is written by the operation.
    @sa         fern::algorithm::nullary_local_operation,
                fern::algorithm::binary_local_operation,
                fern::algorithm::n_ary_local_operation

    This function supports handling 0d, 1d and 2d values.

    This function supports sequential and parallel execution of the operation.
*/
template<
    template<typename> class Algorithm,
    template<typename> class OutOfDomainPolicy,
    template<typename, typename> class OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename ExecutionPolicy,
    typename Value,
    typename Result
>
void unary_local_operation(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy& execution_policy,
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
