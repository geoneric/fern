#pragma once
#include "fern/core/data_traits.h"
#include "fern/core/base_class.h"
#include "fern/feature/core/masked_array_traits.h"
#include "fern/algorithm/core/index_ranges.h"
#include "fern/algorithm/policy/policies.h"


namespace fern {
namespace algorithm {
namespace binary_aggregate_operation_ {
namespace detail {

template<
    typename OutOfDomainPolicy,
    template<typename, typename, typename> class OutOfRangePolicy,
    typename Algorithm,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result>
void operation_0d(
    Algorithm const& algorithm,
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    Value const& values,
    value_type<Value> const& value,
    Result& result)
{
    if(std::get<0>(input_no_data_policy).is_no_data() ||
            std::get<1>(input_no_data_policy).is_no_data()) {
        output_no_data_policy.mark_as_no_data();
    }
    else {
        algorithm.init(get(values), get(value), get(result));
    }
}


template<
    typename OutOfDomainPolicy,
    template<typename, typename, typename> class OutOfRangePolicy,
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
    Value const& values,
    value_type<Value> const& value,
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

        for(size_t i = begin; i < end; ++i) {

            if(!(std::get<0>(input_no_data_policy).is_no_data(i) ||
                    std::get<1>(input_no_data_policy).is_no_data(i))) {

                // Initialize result using the first valid value.
                algorithm.init(get(values, i), get(value),
                    tmp_result);
                result_ = tmp_result;
                data_seen = true;

                for(++i; i < end; ++i) {

                    if(!(std::get<0>(input_no_data_policy).is_no_data(i) ||
                            std::get<1>(input_no_data_policy).is_no_data(i))) {

                        value_type<Value> const& a_value{get(values, i)};
                        algorithm.calculate(a_value, get(value),
                            tmp_result);

                        if(!OORP::within_range(a_value, get(value),
                                tmp_result)) {
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
    typename OutOfDomainPolicy,
    template<typename, typename, typename> class OutOfRangePolicy,
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
    Value const& values,
    value_type<Value> const& value,
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

        // Initialize result.
        for(; i < end1; ++i) {

            index_ = index(values, i, j);

            for(; j < end2; ++j) {

                if(!(std::get<0>(input_no_data_policy).is_no_data(index_) ||
                        std::get<1>(input_no_data_policy).is_no_data(index_))) {

                    // Initialize result using the first valid value.
                    algorithm.init(get(values, index_), get(value),
                        tmp_result);
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

                index_ = index(values, i, j);

                for(; j < end2; ++j) {

                    if(!(std::get<0>(input_no_data_policy).is_no_data(index_) ||
                            std::get<1>(
                                input_no_data_policy).is_no_data(index_))) {

                        value_type<Value> const& a_value{get(
                            values, index_)};
                        algorithm.calculate(a_value, get(value),
                            tmp_result);

                        // lhs, rhs, lhs + rhs
                        if(!OORP::within_range(a_value, get(value),
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
    typename Algorithm,
    typename Aggregator,
    typename OutOfDomainPolicy,
    template<typename, typename, typename> class OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result,
    typename ExecutionPolicy,
    typename ValueCollectionCategory>
struct BinaryAggregateOperation
{
};


template<
    typename Algorithm,
    typename Aggregator,
    typename OutOfDomainPolicy,
    template<typename, typename, typename> class OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result,
    typename ExecutionPolicy>
struct BinaryAggregateOperation<
    Algorithm,
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
        Value const& values,
        value_type<Value> const& value,
        Result& result)
    {
        Algorithm algorithm;

        operation_0d<OutOfDomainPolicy, OutOfRangePolicy>(algorithm,
            input_no_data_policy, output_no_data_policy,
            values, value, result);
    }

};


template<
    typename Algorithm,
    typename Aggregator,
    typename OutOfDomainPolicy,
    template<typename, typename, typename> class OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result>
struct BinaryAggregateOperation<
    Algorithm,
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
        Value const& values,
        value_type<Value> const& value,
        Result& result)
    {
        Algorithm algorithm;

        operation_1d<OutOfDomainPolicy, OutOfRangePolicy>(algorithm,
            input_no_data_policy, output_no_data_policy,
            IndexRanges<1>{IndexRange(0, size(values))},
            values, value, result);
    }

};


template<
    typename Algorithm,
    typename Aggregator,
    typename OutOfDomainPolicy,
    template<typename, typename, typename> class OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result>
struct BinaryAggregateOperation<
    Algorithm,
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
        Value const& values,
        value_type<Value> const& value,
        Result& result)
    {
        // Algorithm algorithm;

        // operation_1d<OutOfDomainPolicy, OutOfRangePolicy>(algorithm,
        //     input_no_data_policy, output_no_data_policy,
        //     IndexRanges<1>{IndexRange(0, size(values))},
        //     values, value, result);

        ThreadPool& pool(execution_policy.thread_pool());
        size_t const size_ = size(values);
        std::vector<IndexRanges<1>> ranges = index_ranges(pool.size(), size_);
        std::vector<std::future<void>> futures;
        futures.reserve(ranges.size());
        std::vector<Result> results_per_block(ranges.size(), Result(0));

        Algorithm algorithm;

        // TODO OutputNoDataPolicy must be created ad-hoc here. It must have
        //      an effect on the values in results_per_block.
        //      This depends on whether or not results are masking. Probably
        //      need to dispatch this whole function on that. First create a
        //      failing unit test that fails: include a block with only
        //      no-data and verify the result is not no-data.
        for(size_t i = 0; i < ranges.size(); ++i) {
            auto const& block_range(ranges[i]);
            auto function = std::bind(
                operation_1d<OutOfDomainPolicy, OutOfRangePolicy, Algorithm,
                    InputNoDataPolicy, OutputNoDataPolicy,
                    Value, Result>,
                std::cref(algorithm),
                std::cref(input_no_data_policy),
                std::ref(output_no_data_policy), std::cref(block_range),
                std::cref(values), std::cref(value),
                std::ref(results_per_block[i]));
            futures.emplace_back(pool.submit(function));
        }

        for(auto& future: futures) {
            future.get();
        }

        Aggregator aggregator;

        Aggregate<DataTraits<Result>::is_masking>::template
            apply<OutOfRangePolicy>(
                aggregator, output_no_data_policy, execution_policy,
                results_per_block, result);
    }

};


template<
    typename Algorithm,
    typename Aggregator,
    typename OutOfDomainPolicy,
    template<typename, typename, typename> class OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result>
struct BinaryAggregateOperation<
    Algorithm,
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
        Value const& values,
        value_type<Value> const& value,
        Result& result)
    {
        switch(execution_policy.which()) {
            case fern::algorithm::detail::sequential_execution_policy_id: {
                detail::dispatch::BinaryAggregateOperation<
                    Algorithm,
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
                        values, value, result);
                break;
            }
            case fern::algorithm::detail::parallel_execution_policy_id: {
                detail::dispatch::BinaryAggregateOperation<
                    Algorithm,
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
                        values, value, result);
                break;
            }
        }
    }

};


template<
    typename Algorithm,
    typename Aggregator,
    typename OutOfDomainPolicy,
    template<typename, typename, typename> class OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result>
struct BinaryAggregateOperation<
    Algorithm,
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
        Value const& values,
        value_type<Value> const& value,
        Result& result)
    {
        Algorithm algorithm;

        operation_2d<OutOfDomainPolicy, OutOfRangePolicy>(algorithm,
            input_no_data_policy, output_no_data_policy,
            IndexRanges<2>{
                IndexRange(0, size(values, 0)),
                IndexRange(0, size(values, 1))},
            values, value, result);
    }

};


template<
    typename Algorithm,
    typename Aggregator,
    typename OutOfDomainPolicy,
    template<typename, typename, typename> class OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result>
struct BinaryAggregateOperation<
    Algorithm,
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
        Value const& values,
        value_type<Value> const& value,
        Result& result)
    {
        ThreadPool& pool(execution_policy.thread_pool());
        size_t const size1 = size(values, 0);
        size_t const size2 = size(values, 1);
        std::vector<IndexRanges<2>> ranges = index_ranges(pool.size(),
            size1, size2);
        std::vector<std::future<void>> futures;
        futures.reserve(ranges.size());
        std::vector<Result> results_per_block(ranges.size(), Result(0));

        Algorithm algorithm;

        // TODO OutputNoDataPolicy must be created ad-hoc here. It must have
        //      an effect on the values in results_per_block.
        //      This depends on whether or not results are masking. Probably
        //      need to dispatch this whole function on that. First create a
        //      failing unit test that fails: include a block with only
        //      no-data and verify the result is not no-data.
        for(size_t i = 0; i < ranges.size(); ++i) {
            auto const& block_range(ranges[i]);
            auto function = std::bind(
                operation_2d<OutOfDomainPolicy, OutOfRangePolicy, Algorithm,
                    InputNoDataPolicy, OutputNoDataPolicy,
                    Value, Result>,
                std::cref(algorithm),
                std::cref(input_no_data_policy),
                std::ref(output_no_data_policy), std::cref(block_range),
                std::cref(values), std::cref(value),
                std::ref(results_per_block[i]));
            futures.emplace_back(pool.submit(function));
        }

        for(auto& future: futures) {
            future.get();
        }

        Aggregator aggregator;

        Aggregate<DataTraits<Result>::is_masking>::template
            apply<OutOfRangePolicy>(
                aggregator, output_no_data_policy, execution_policy,
                results_per_block, result);
    }

};


template<
    typename Algorithm,
    typename Aggregator,
    typename OutOfDomainPolicy,
    template<typename, typename, typename> class OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result>
struct BinaryAggregateOperation<
    Algorithm,
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
        Value const& values,
        value_type<Value> const& value,
        Result& result)
    {
        switch(execution_policy.which()) {
            case fern::algorithm::detail::sequential_execution_policy_id: {
                detail::dispatch::BinaryAggregateOperation<
                    Algorithm,
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
                        values, value, result);
                break;
            }
            case fern::algorithm::detail::parallel_execution_policy_id: {
                detail::dispatch::BinaryAggregateOperation<
                    Algorithm,
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
                        values, value, result);
                break;
            }
        }
    }

};

} // namespace dispatch
} // namespace detail
} // namespace binary_aggregate_operation_


/*!
    @ingroup   fern_algorithm_core_group
*/
template<
    template<typename> class Algorithm,
    typename Aggregator,
    template<typename, typename> class OutOfDomainPolicy,
    template<typename, typename, typename> class OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename ExecutionPolicy,
    typename Value,
    typename Result
>
void binary_aggregate_operation(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy& execution_policy,
    Value const& values,
    value_type<Value> const& value,
    Result& result)
{
    binary_aggregate_operation_::detail::dispatch::BinaryAggregateOperation<
        Algorithm<value_type<Value>>,
        Aggregator,
        OutOfDomainPolicy<value_type<Value>, value_type<Value>>,
        OutOfRangePolicy,
        InputNoDataPolicy,
        OutputNoDataPolicy,
        Value,
        Result,
        ExecutionPolicy,
        base_class<argument_category<Value>, array_2d_tag>>::apply(
            input_no_data_policy, output_no_data_policy, execution_policy,
            values, value, result);
}

} // namespace algorithm
} // namespace fern
