#pragma once
#include "fern/core/argument_traits.h"
#include "fern/core/base_class.h"
#include "fern/feature/core/masked_array_traits.h"
#include "fern/algorithm/core/index_ranges.h"
#include "fern/algorithm/policy/policies.h"


namespace fern {
namespace algorithm {
namespace unary_aggregate_operation_ {
namespace detail {

template<
    template<typename, typename> class Accumulator,
    typename OutOfDomainPolicy,
    template<typename, typename, typename> class OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result>
void operation_0d(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    Value const& value,
    Result& result)
{
    if(std::get<0>(input_no_data_policy).is_no_data()) {
        output_no_data_policy.mark_as_no_data();
    }
    else {
        Accumulator<value_type<Value>, value_type<Result>> accumulator(
            get(value));
        get(result) = accumulator();
    }
}


template<
    template<typename, typename> class Accumulator,
    typename OutOfDomainPolicy,
    template<typename, typename, typename> class OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result>
void operation_1d(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    IndexRanges<1> const& index_ranges,
    Value const& value,
    Result& result)
{
    using OORP = OutOfRangePolicy<value_type<Value>, value_type<Value>,
        value_type<Result>>;

    size_t const begin = index_ranges[0].begin();
    size_t const end = index_ranges[0].end();
    bool data_seen{false};

    if(begin < end) {

        value_type<Result> tmp_result;
        value_type<Result>& result_ = get(result);

        Accumulator<value_type<Value>, value_type<Result>> accumulator;

        for(size_t i = begin; i < end; ++i) {

            if(!std::get<0>(input_no_data_policy).is_no_data(i)) {

                // Initialize result using the first valid value.
                accumulator = get(value, i);
                tmp_result = accumulator();
                result_ = tmp_result;
                data_seen = true;

                for(++i; i < end; ++i) {

                    if(!std::get<0>(input_no_data_policy).is_no_data(i)) {

                        value_type<Value> const& a_value{get(value, i)};
                        accumulator(a_value);
                        tmp_result = accumulator();

                        // lhs, rhs, lhs + rhs
                        if(!OORP::within_range(result_, a_value, tmp_result)) {
                            output_no_data_policy.mark_as_no_data();
                            break;
                        }

                        result_ = tmp_result;
                    }
                }
            }
        }
    }

    if(!data_seen) {
        output_no_data_policy.mark_as_no_data();
    }
}


template<
    template<typename, typename> class Accumulator,
    typename OutOfDomainPolicy,
    template<typename, typename, typename> class OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result>
void operation_2d(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    IndexRanges<2> const& index_ranges,
    Value const& value,
    Result& result)
{
    using OORP = OutOfRangePolicy<value_type<Value>, value_type<Value>,
        value_type<Result>>;

    size_t const begin1 = index_ranges[0].begin();
    size_t const end1 = index_ranges[0].end();
    size_t const begin2 = index_ranges[1].begin();
    size_t const end2 = index_ranges[1].end();
    bool data_seen{false};

    if(begin1 < end1 && begin2 < end2) {

        value_type<Result> tmp_result;
        value_type<Result>& result_ = get(result);

        size_t i = begin1;
        size_t j = begin2;
        size_t index_;

        Accumulator<value_type<Value>, value_type<Result>> accumulator;

        // Initialize result.
        for(; i < end1; ++i) {

            index_ = index(value, i, j);

            for(; j < end2; ++j) {

                if(!std::get<0>(input_no_data_policy).is_no_data(index_)) {

                    // Initialize result using the first valid value.
                    accumulator = get(value, index_);
                    tmp_result = accumulator();
                    result_ = tmp_result;
                    data_seen = true;
                    break;
                }

                ++index_;
            }

            if(data_seen) {
                break;
            }
        }

        // Continue where the previous loop stopped.
        if(data_seen) {
            ++j;

            for(; i < end1; ++i) {

                index_ = index(value, i, j);

                for(; j < end2; ++j) {

                    if(!std::get<0>(input_no_data_policy).is_no_data(index_)) {

                        value_type<Value> const& a_value{get(value, index_)};
                        accumulator(a_value);
                        tmp_result = accumulator();

                        // lhs, rhs, lhs + rhs
                        if(!OORP::within_range(result_, a_value,
                                tmp_result)) {
                            output_no_data_policy.mark_as_no_data();
                            break;
                        }

                        result_ = tmp_result;
                    }

                    ++index_;
                }

                if(j != end2) {
                    // This happens if the inner loop calls break.
                    // Set i and j such that all loops exit.
                    i = end1;
                    j = end2;
                }
                else {
                    j = begin2;
                }
            }
        }
    }

    if(!data_seen) {
        output_no_data_policy.mark_as_no_data();
    }
}


namespace dispatch {

template<
    bool result_is_masking>
struct Aggregate
{
};


template<>
struct Aggregate<
    false>
{

    template<
        template<typename, typename, typename> class OutOfRangePolicy,
        typename Aggregator,
        typename OutputNoDataPolicy,
        typename ExecutionPolicy,
        typename Results,
        typename Result>
    static void apply(
        Aggregator const& aggregator,
        OutputNoDataPolicy& output_no_data_policy,
        ExecutionPolicy& /* execution_policy */,
        Results const& results,
        Result& result)
    {
        Array<value_type<Result>, 1> results_(results);

        // Accumulate the results into one single result.
        // The final result is not masking, so the results aren't
        // either.
        aggregator.template apply<OutOfRangePolicy>(
            InputNoDataPolicies<SkipNoData>{{}}, output_no_data_policy,
            sequential, results_, result);
    }

};


template<>
struct Aggregate<
    true>
{

    template<
        template<typename, typename, typename> class OutOfRangePolicy,
        typename Aggregator,
        typename OutputNoDataPolicy,
        typename ExecutionPolicy,
        typename Results,
        typename Result>
    static void apply(
        Aggregator const& aggregator,
        OutputNoDataPolicy& output_no_data_policy,
        ExecutionPolicy& /* execution_policy */,
        Results const& results,
        Result& result)
    {
        MaskedArray<value_type<Result>, 1> results_(results);

        // Accumulate the results into one single result.
        // The final result is masking, so the results are also.
        aggregator.template apply<OutOfRangePolicy>(
            InputNoDataPolicies<DetectNoDataByValue<Mask<1>>>{{results_.mask(),
                true}}, output_no_data_policy, sequential, results_, result);
    }

};


template<
    template<typename, typename> class Accumulator,
    typename Aggregator,
    typename OutOfDomainPolicy,
    template<typename, typename, typename> class OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result,
    typename ExecutionPolicy,
    typename ValueCollectionCategory>
struct UnaryAggregateOperation
{
};


template<
    template<typename, typename> class Accumulator,
    typename Aggregator,
    typename OutOfDomainPolicy,
    template<typename, typename, typename> class OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result,
    typename ExecutionPolicy>
struct UnaryAggregateOperation<
    Accumulator,
    Aggregator,
    OutOfDomainPolicy,
    OutOfRangePolicy,
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value,
    Result,
    ExecutionPolicy,
    array_0d_tag>
{

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ExecutionPolicy& /* execution_policy */,
        Value const& value,
        Result& result)
    {
        operation_0d<Accumulator, OutOfDomainPolicy, OutOfRangePolicy>(
            input_no_data_policy, output_no_data_policy,
            value, result);
    }

};


template<
    template<typename, typename> class Accumulator,
    typename Aggregator,
    typename OutOfDomainPolicy,
    template<typename, typename, typename> class OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result>
struct UnaryAggregateOperation<
    Accumulator,
    Aggregator,
    OutOfDomainPolicy,
    OutOfRangePolicy,
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value,
    Result,
    SequentialExecutionPolicy,
    array_1d_tag>
{

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SequentialExecutionPolicy& /* execution_policy */,
        Value const& value,
        Result& result)
    {
        operation_1d<Accumulator, OutOfDomainPolicy, OutOfRangePolicy>(
            input_no_data_policy, output_no_data_policy,
            IndexRanges<1>{IndexRange(0, size(value))},
            value, result);
    }

};


template<
    template<typename, typename> class Accumulator,
    typename Aggregator,
    typename OutOfDomainPolicy,
    template<typename, typename, typename> class OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result>
struct UnaryAggregateOperation<
    Accumulator,
    Aggregator,
    OutOfDomainPolicy,
    OutOfRangePolicy,
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value,
    Result,
    ParallelExecutionPolicy,
    array_1d_tag>
{

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ParallelExecutionPolicy& execution_policy,
        Value const& value,
        Result& result)
    {
        ThreadPool& pool(execution_policy.thread_pool());
        size_t const size_ = size(value);
        std::vector<IndexRanges<1>> ranges = index_ranges(pool.size(), size_);
        std::vector<std::future<void>> futures;
        futures.reserve(ranges.size());
        std::vector<Result> results_per_block(ranges.size(), Result(0));

        // TODO OutputNoDataPolicy must be created ad-hoc here. It must have
        //      an effect on the values in results_per_block.
        //      This depends on whether or not results are masking. Probably
        //      need to dispatch this whole function on that. First create a
        //      failing unit test that fails: include a block with only
        //      no-data and verify the result is not no-data.
        for(size_t i = 0; i < ranges.size(); ++i) {
            auto const& block_range(ranges[i]);
            auto function = std::bind(
                operation_1d<Accumulator, OutOfDomainPolicy, OutOfRangePolicy,
                    InputNoDataPolicy, OutputNoDataPolicy,
                    Value, Result>,
                std::cref(input_no_data_policy),
                std::ref(output_no_data_policy), std::cref(block_range),
                std::cref(value), std::ref(results_per_block[i]));
            futures.emplace_back(pool.submit(function));
        }

        for(auto& future: futures) {
            future.get();
        }

        Aggregator aggregator;

        Aggregate<ArgumentTraits<Result>::is_masking>::template
            apply<OutOfRangePolicy>(
                aggregator, output_no_data_policy, execution_policy,
                results_per_block, result);
    }

};


template<
    template<typename, typename> class Accumulator,
    typename Aggregator,
    typename OutOfDomainPolicy,
    template<typename, typename, typename> class OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result>
struct UnaryAggregateOperation<
    Accumulator,
    Aggregator,
    OutOfDomainPolicy,
    OutOfRangePolicy,
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value,
    Result,
    ExecutionPolicy,
    array_1d_tag>
{

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ExecutionPolicy& execution_policy,
        Value const& value,
        Result& result)
    {
        switch(execution_policy.which()) {
            case fern::algorithm::detail::sequential_execution_policy_id: {
                UnaryAggregateOperation<
                    Accumulator,
                    Aggregator,
                    OutOfDomainPolicy,
                    OutOfRangePolicy,
                    InputNoDataPolicy,
                    OutputNoDataPolicy,
                    Value,
                    Result,
                    SequentialExecutionPolicy,
                    array_1d_tag>::apply(
                        input_no_data_policy, output_no_data_policy,
                        boost::get<SequentialExecutionPolicy>(execution_policy),
                        value, result);
                break;
            }
            case fern::algorithm::detail::parallel_execution_policy_id: {
                UnaryAggregateOperation<
                    Accumulator,
                    Aggregator,
                    OutOfDomainPolicy,
                    OutOfRangePolicy,
                    InputNoDataPolicy,
                    OutputNoDataPolicy,
                    Value,
                    Result,
                    ParallelExecutionPolicy,
                    array_1d_tag>::apply(
                        input_no_data_policy, output_no_data_policy,
                        boost::get<ParallelExecutionPolicy>(execution_policy),
                        value, result);
                break;
            }
        }
    }

};


template<
    template<typename, typename> class Accumulator,
    typename Aggregator,
    typename OutOfDomainPolicy,
    template<typename, typename, typename> class OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result>
struct UnaryAggregateOperation<
    Accumulator,
    Aggregator,
    OutOfDomainPolicy,
    OutOfRangePolicy,
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value,
    Result,
    SequentialExecutionPolicy,
    array_2d_tag>
{

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SequentialExecutionPolicy& /* execution_policy */,
        Value const& value,
        Result& result)
    {
        operation_2d<Accumulator, OutOfDomainPolicy, OutOfRangePolicy>(
            input_no_data_policy, output_no_data_policy,
            IndexRanges<2>{
                IndexRange(0, size(value, 0)),
                IndexRange(0, size(value, 1))},
            value, result);
    }

};


template<
    template<typename, typename> class Accumulator,
    typename Aggregator,
    typename OutOfDomainPolicy,
    template<typename, typename, typename> class OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result>
struct UnaryAggregateOperation<
    Accumulator,
    Aggregator,
    OutOfDomainPolicy,
    OutOfRangePolicy,
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value,
    Result,
    ParallelExecutionPolicy,
    array_2d_tag>
{

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ParallelExecutionPolicy& execution_policy,
        Value const& value,
        Result& result)
    {
        ThreadPool& pool(execution_policy.thread_pool());
        size_t const size1 = size(value, 0);
        size_t const size2 = size(value, 1);
        std::vector<IndexRanges<2>> ranges = index_ranges(pool.size(),
            size1, size2);
        std::vector<std::future<void>> futures;
        futures.reserve(ranges.size());
        std::vector<Result> results_per_block(ranges.size(), Result(0));

        // TODO OutputNoDataPolicy must be created ad-hoc here. It must have
        //      an effect on the values in results_per_block.
        //      This depends on whether or not results are masking. Probably
        //      need to dispatch this whole function on that. First create a
        //      failing unit test that fails: include a block with only
        //      no-data and verify the result is not no-data.
        for(size_t i = 0; i < ranges.size(); ++i) {
            auto const& block_range(ranges[i]);
            auto function = std::bind(
                operation_2d<Accumulator, OutOfDomainPolicy, OutOfRangePolicy,
                    InputNoDataPolicy, OutputNoDataPolicy,
                    Value, Result>,
                std::cref(input_no_data_policy),
                std::ref(output_no_data_policy), std::cref(block_range),
                std::cref(value), std::ref(results_per_block[i]));
            futures.emplace_back(pool.submit(function));
        }

        for(auto& future: futures) {
            future.get();
        }

        Aggregator aggregator;

        Aggregate<ArgumentTraits<Result>::is_masking>::template
            apply<OutOfRangePolicy>(
                aggregator, output_no_data_policy, execution_policy,
                results_per_block, result);
    }

};


template<
    template<typename, typename> class Accumulator,
    typename Aggregator,
    typename OutOfDomainPolicy,
    template<typename, typename, typename> class OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result>
struct UnaryAggregateOperation<
    Accumulator,
    Aggregator,
    OutOfDomainPolicy,
    OutOfRangePolicy,
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value,
    Result,
    ExecutionPolicy,
    array_2d_tag>
{

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ExecutionPolicy& execution_policy,
        Value const& value,
        Result& result)
    {
        switch(execution_policy.which()) {
            case fern::algorithm::detail::sequential_execution_policy_id: {
                UnaryAggregateOperation<
                    Accumulator,
                    Aggregator,
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
            case fern::algorithm::detail::parallel_execution_policy_id: {
                UnaryAggregateOperation<
                    Accumulator,
                    Aggregator,
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
} // namespace unary_aggregate_operation_


/*!
    @ingroup    fern_algorithm_core_group
    @brief      Function that executes a unary aggregate operation.
    @tparam     Accumulator Class template of the operation to execute.
    @tparam     Aggregator Class of the aggregator.
    @param[in]  value Input to pass to the operation.
    @param[out] result Output that is written by the operation.
    @sa         fern::algorithm::binary_aggregate_operation

    This function supports handling 0d, 1d and 2d values.

    This function supports sequential and parallel execution of the operation.
*/
template<
    template<typename, typename> class Accumulator,
    typename Aggregator,
    template<typename> class OutOfDomainPolicy,
    template<typename, typename, typename> class OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename ExecutionPolicy,
    typename Value,
    typename Result
>
void unary_aggregate_operation(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy& execution_policy,
    Value const& value,
    Result& result)
{
    unary_aggregate_operation_::detail::dispatch::UnaryAggregateOperation<
        Accumulator,
        Aggregator,
        OutOfDomainPolicy<value_type<Value>>,
        OutOfRangePolicy,
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
