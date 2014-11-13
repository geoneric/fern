#pragma once
#include "fern/core/base_class.h"
#include "fern/core/thread_client.h"
#include "fern/algorithm/core/index_ranges.h"


namespace fern {
namespace algorithm {
namespace unite_no_data {
namespace detail {

template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value1,
    typename Value2,
    typename Result>
void unite_no_data_0d_0d(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    Value1 const& /* value1 */,
    Value2 const& /* value2 */,
    Result& /* result */)
{
    if(!input_no_data_policy.is_no_data()) {
        if(input_no_data_policy.get<0>().is_no_data() ||
                input_no_data_policy.get<1>().is_no_data()) {
            output_no_data_policy.mark_as_no_data();
        }
    }
}


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value1,
    typename Value2,
    typename Result>
void unite_no_data_2d_2d(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    IndexRanges<2> const& index_ranges,
    Value1 const& /* value1 */,
    Value2 const& /* value2 */,
    Result& /* result */)
{
    for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end(); ++i) {
        for(size_t j = index_ranges[1].begin(); j < index_ranges[1].end();
                ++j) {

            if(!input_no_data_policy.is_no_data(i, j)) {
                if(input_no_data_policy.get<0>().is_no_data(i, j) ||
                        input_no_data_policy.get<1>().is_no_data(i, j)) {
                    output_no_data_policy.mark_as_no_data(i, j);
                }
            }
        }
    }
}


namespace dispatch {

template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value1,
    typename Value2,
    typename Result,
    typename ExecutionPolicy,
    typename Value1CollectionCategory,
    typename Value2CollectionCategory>
struct UniteNoDataByArgumentCategory
{
};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value1,
    typename Value2,
    typename Result,
    typename ExecutionPolicy>
struct UniteNoDataByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value1,
    Value2,
    Result,
    ExecutionPolicy,
    array_0d_tag,
    array_0d_tag>
{

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ExecutionPolicy const& /* execution_policy */,
        Value1 const& value1,
        Value2 const& value2,
        Result& result)
    {
        unite_no_data_0d_0d(input_no_data_policy, output_no_data_policy,
            value1, value2, result);
    }

};


// TODO: 1d stuff.


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value1,
    typename Value2,
    typename Result>
struct UniteNoDataByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value1,
    Value2,
    Result,
    SequentialExecutionPolicy,
    array_2d_tag,
    array_2d_tag>
{

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
        assert(size(value2, 0) == size(result, 0));
        assert(size(value2, 1) == size(result, 1));

        unite_no_data_2d_2d(input_no_data_policy, output_no_data_policy,
            IndexRanges<2>{
                IndexRange(0, size(result, 0)),
                IndexRange(0, size(result, 1)),
            }, value1, value2, result);
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value1,
    typename Value2,
    typename Result>
struct UniteNoDataByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value1,
    Value2,
    Result,
    ParallelExecutionPolicy,
    array_2d_tag,
    array_2d_tag>
{

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
        assert(size(value2, 0) == size(result, 0));
        assert(size(value2, 1) == size(result, 1));

        ThreadPool& pool(ThreadClient::pool());
        size_t const size1 = size(result, 0);
        size_t const size2 = size(result, 1);
        std::vector<IndexRanges<2>> ranges = index_ranges(pool.size(),
            size1, size2);
        std::vector<std::future<void>> futures;
        futures.reserve(ranges.size());

        for(auto const& block_range: ranges) {
            auto function = std::bind(
                unite_no_data_2d_2d<
                    InputNoDataPolicy, OutputNoDataPolicy,
                    Value1, Value2, Result>,
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
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value1,
    typename Value2,
    typename Result,
    typename ExecutionPolicy>
class UniteNoDataByExecutionPolicy
{
};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value1,
    typename Value2,
    typename Result>
struct UniteNoDataByExecutionPolicy<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value1,
    Value2,
    Result,
    ExecutionPolicy>
{

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ExecutionPolicy const& execution_policy,
        Value1 const& value1,
        Value2 const& value2,
        Result& result)
    {
        switch(execution_policy.which()) {
            case fern::algorithm::detail::sequential_execution_policy_id: {
                UniteNoDataByArgumentCategory<
                    InputNoDataPolicy,
                    OutputNoDataPolicy,
                    Value1,
                    Value2,
                    Result,
                    SequentialExecutionPolicy,
                    base_class<argument_category<Value1>, array_2d_tag>,
                    base_class<argument_category<Value2>, array_2d_tag>>
                        ::apply(
                            input_no_data_policy, output_no_data_policy,
                            fern::algorithm::detail::get_policy<
                                SequentialExecutionPolicy>(execution_policy),
                            value1, value2, result);
                break;
            }
            case fern::algorithm::detail::parallel_execution_policy_id: {
                UniteNoDataByArgumentCategory<
                    InputNoDataPolicy,
                    OutputNoDataPolicy,
                    Value1,
                    Value2,
                    Result,
                    ParallelExecutionPolicy,
                    base_class<argument_category<Value1>, array_2d_tag>,
                    base_class<argument_category<Value2>, array_2d_tag>>
                        ::apply(
                            input_no_data_policy, output_no_data_policy,
                            fern::algorithm::detail::get_policy<
                                ParallelExecutionPolicy>(execution_policy),
                            value1, value2, result);
                break;
            }
        }
    }

};

} // namespace dispatch


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename ExecutionPolicy,
    typename Value1,
    typename Value2,
    typename Result>
void unite_no_data(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy const& execution_policy,
    Value1 const& value1,
    Value2 const& value2,
    Result& result)
{
    dispatch::UniteNoDataByExecutionPolicy<InputNoDataPolicy,
        OutputNoDataPolicy, Value1, Value2, Result, ExecutionPolicy>::
            apply(input_no_data_policy, output_no_data_policy,
                execution_policy, value1, value2, result);
}

} // namespace detail
} // namespace unite_no_data
} // namespace algorithm
} // namespace fern
