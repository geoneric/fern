#pragma once
#include "fern/core/array_2d_traits.h"
#include "fern/core/base_class.h"
#include "fern/core/constant_traits.h"
#include "fern/core/thread_client.h"
#include "fern/algorithm/core/index_ranges.h"
#include "fern/algorithm/policy/execution_policy.h"


namespace fern {
namespace algorithm {
namespace if_ {
namespace detail {

template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Condition,
    class TrueValue,
    class Result>
void if_then_0d_0d(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    Condition const& condition,
    TrueValue const& true_value,
    Result& result)
{
    if(!input_no_data_policy.is_no_data()) {
        if(fern::get(condition)) {
            fern::get(result) = fern::get(true_value);
        }
        else {
            output_no_data_policy.mark_as_no_data();
        }
    }
}


template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Condition,
    class TrueValue,
    class Result>
void if_then_2d_0d(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    IndexRanges<2> const& index_ranges,
    Condition const& condition,
    TrueValue const& true_value,
    Result& result)
{
    for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end(); ++i) {
        for(size_t j = index_ranges[1].begin(); j < index_ranges[1].end();
                ++j) {

            if(!input_no_data_policy.is_no_data(i, j)) {

                if(fern::get(condition, i, j)) {
                    fern::get(result, i, j) = fern::get(true_value);
                }
                else {
                    output_no_data_policy.mark_as_no_data(i, j);
                }
            }
        }
    }
}


template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Condition,
    class TrueValue,
    class Result>
void if_then_2d_2d(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    IndexRanges<2> const& index_ranges,
    Condition const& condition,
    TrueValue const& true_value,
    Result& result)
{
    for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end(); ++i) {
        for(size_t j = index_ranges[1].begin(); j < index_ranges[1].end();
                ++j) {

            if(!input_no_data_policy.is_no_data(i, j)) {

                if(fern::get(condition, i, j)) {
                    fern::get(result, i, j) = fern::get(true_value, i, j);
                }
                else {
                    output_no_data_policy.mark_as_no_data(i, j);
                }
            }
        }
    }
}


template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Condition,
    class TrueValue,
    class FalseValue,
    class Result>
void if_then_else_0d_0d_0d(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& /* output_no_data_policy */,
    Condition const& condition,
    TrueValue const& true_value,
    FalseValue const& false_value,
    Result& result)
{
    if(!input_no_data_policy.is_no_data()) {
        if(fern::get(condition)) {
            fern::get(result) = fern::get(true_value);
        }
        else {
            fern::get(result) = fern::get(false_value);
        }
    }
}


template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Condition,
    class TrueValue,
    class FalseValue,
    class Result>
void if_then_else_2d_2d_2d(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& /* output_no_data_policy */,
    IndexRanges<2> const& index_ranges,
    Condition const& condition,
    TrueValue const& true_value,
    FalseValue const& false_value,
    Result& result)
{
    for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end(); ++i) {
        for(size_t j = index_ranges[1].begin(); j < index_ranges[1].end();
                ++j) {

            if(!input_no_data_policy.is_no_data(i, j)) {

                if(fern::get(condition, i, j)) {
                    fern::get(result, i, j) = fern::get(true_value, i, j);
                }
                else {
                    fern::get(result, i, j) = fern::get(false_value, i, j);
                }
            }
        }
    }
}


template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Condition,
    class TrueValue,
    class FalseValue,
    class Result>
void if_then_else_2d_0d_0d(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& /* output_no_data_policy */,
    IndexRanges<2> const& index_ranges,
    Condition const& condition,
    TrueValue const& true_value,
    FalseValue const& false_value,
    Result& result)
{
    for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end(); ++i) {
        for(size_t j = index_ranges[1].begin(); j < index_ranges[1].end();
                ++j) {

            if(!input_no_data_policy.is_no_data(i, j)) {

                if(fern::get(condition, i, j)) {
                    fern::get(result, i, j) = fern::get(true_value);
                }
                else {
                    fern::get(result, i, j) = fern::get(false_value);
                }
            }
        }
    }
}


namespace dispatch {

template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Condition,
    class TrueValue,
    class Result,
    class ExecutionPolicy,
    class ConditionCollectionCategory,
    class TrueValueCollectionCategory>
class IfThenByArgumentCategory
{
};


template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Condition,
    class TrueValue,
    class FalseValue,
    class Result,
    class ExecutionPolicy,
    class ConditionCollectionCategory,
    class TrueValueCollectionCategory,
    class FalseValueCollectionCategory>
struct IfThenElseByArgumentCategory
{
};


template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Condition,
    class TrueValue,
    class Result,
    class ExecutionPolicy>
struct IfThenByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Condition,
    TrueValue,
    Result,
    ExecutionPolicy,
    array_0d_tag,
    array_0d_tag>
{

    // if(0d, 0d, 0d)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ExecutionPolicy const& /* execution_policy */,
        Condition const& condition,
        TrueValue const& true_value,
        Result& result)
    {
        if_then_0d_0d(input_no_data_policy, output_no_data_policy,
            condition, true_value, result);
    }

};


template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Condition,
    class TrueValue,
    class FalseValue,
    class Result,
    class ExecutionPolicy>
struct IfThenElseByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Condition,
    TrueValue,
    FalseValue,
    Result,
    ExecutionPolicy,
    array_0d_tag,
    array_0d_tag,
    array_0d_tag>
{

    // if(0d, 0d, 0d, 0d)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ExecutionPolicy const& /* execution_policy */,
        Condition const& condition,
        TrueValue const& true_value,
        FalseValue const& false_value,
        Result& result)
    {
        if_then_else_0d_0d_0d(input_no_data_policy, output_no_data_policy,
            condition, true_value, false_value, result);
    }

};


// TODO: 1d stuff.


template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Condition,
    class TrueValue,
    class Result>
struct IfThenByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Condition,
    TrueValue,
    Result,
    SequentialExecutionPolicy,
    array_2d_tag,
    array_0d_tag>
{

    // if(2d, 0d, 2d)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SequentialExecutionPolicy const& /* execution_policy */,
        Condition const& condition,
        TrueValue const& true_value,
        Result& result)
    {
        assert(fern::size(result, 0) == fern::size(condition, 0));
        assert(fern::size(result, 1) == fern::size(condition, 1));

        if_then_2d_0d(input_no_data_policy, output_no_data_policy,
            IndexRanges<2>{
                IndexRange(0, fern::size(condition, 0)),
                IndexRange(0, fern::size(condition, 1)),
            }, condition, true_value, result);
    }

};


template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Condition,
    class TrueValue,
    class Result>
struct IfThenByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Condition,
    TrueValue,
    Result,
    ParallelExecutionPolicy,
    array_2d_tag,
    array_0d_tag>
{

    // if(2d, 0d, 2d)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ParallelExecutionPolicy const& /* execution_policy */,
        Condition const& condition,
        TrueValue const& true_value,
        Result& result)
    {
        assert(fern::size(result, 0) == fern::size(condition, 0));
        assert(fern::size(result, 1) == fern::size(condition, 1));

        ThreadPool& pool(ThreadClient::pool());
        size_t const size1 = fern::size(condition, 0);
        size_t const size2 = fern::size(condition, 1);
        std::vector<IndexRanges<2>> ranges = index_ranges(pool.size(),
            size1, size2);
        std::vector<std::future<void>> futures;
        futures.reserve(ranges.size());

        for(auto const& block_range: ranges) {
            auto function = std::bind(
                if_then_2d_0d<
                    InputNoDataPolicy, OutputNoDataPolicy,
                    Condition, TrueValue, Result>,
                std::cref(input_no_data_policy),
                std::ref(output_no_data_policy), std::cref(block_range),
                std::cref(condition), std::cref(true_value), std::ref(result));
            futures.emplace_back(pool.submit(function));
        }

        for(auto& future: futures) {
            future.get();
        }
    }

};


template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Condition,
    class TrueValue,
    class FalseValue,
    class Result>
struct IfThenElseByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Condition,
    TrueValue,
    FalseValue,
    Result,
    SequentialExecutionPolicy,
    array_2d_tag,
    array_0d_tag,
    array_0d_tag>
{

    // if(2d, 0d, 0d, 2d)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SequentialExecutionPolicy const& /* execution_policy */,
        Condition const& condition,
        TrueValue const& true_value,
        FalseValue const& false_value,
        Result& result)
    {
        assert(fern::size(result, 0) == fern::size(condition, 0));
        assert(fern::size(result, 1) == fern::size(condition, 1));

        if_then_else_2d_0d_0d(input_no_data_policy, output_no_data_policy,
            IndexRanges<2>{
                IndexRange(0, fern::size(condition, 0)),
                IndexRange(0, fern::size(condition, 1)),
            }, condition, true_value, false_value, result);
    }

};


template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Condition,
    class TrueValue,
    class FalseValue,
    class Result>
struct IfThenElseByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Condition,
    TrueValue,
    FalseValue,
    Result,
    ParallelExecutionPolicy,
    array_2d_tag,
    array_0d_tag,
    array_0d_tag>
{

    // if(2d, 0d, 0d, 2d)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ParallelExecutionPolicy const& /* execution_policy */,
        Condition const& condition,
        TrueValue const& true_value,
        FalseValue const& false_value,
        Result& result)
    {
        assert(fern::size(result, 0) == fern::size(condition, 0));
        assert(fern::size(result, 1) == fern::size(condition, 1));

        ThreadPool& pool(ThreadClient::pool());
        size_t const size1 = fern::size(condition, 0);
        size_t const size2 = fern::size(condition, 1);
        std::vector<IndexRanges<2>> ranges = index_ranges(pool.size(),
            size1, size2);
        std::vector<std::future<void>> futures;
        futures.reserve(ranges.size());

        for(auto const& block_range: ranges) {
            auto function = std::bind(
                if_then_else_2d_0d_0d<
                    InputNoDataPolicy, OutputNoDataPolicy,
                    Condition, TrueValue, FalseValue, Result>,
                std::cref(input_no_data_policy),
                std::ref(output_no_data_policy), std::cref(block_range),
                std::cref(condition), std::cref(true_value),
                std::cref(false_value), std::ref(result));
            futures.emplace_back(pool.submit(function));
        }

        for(auto& future: futures) {
            future.get();
        }
    }

};


template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Condition,
    class TrueValue,
    class Result>
struct IfThenByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Condition,
    TrueValue,
    Result,
    SequentialExecutionPolicy,
    array_2d_tag,
    array_2d_tag>
{

    // if(2d, 2d, 2d)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SequentialExecutionPolicy const& /* execution_policy */,
        Condition const& condition,
        TrueValue const& true_value,
        Result& result)
    {
        assert(fern::size(true_value, 0) == fern::size(condition, 0));
        assert(fern::size(true_value, 1) == fern::size(condition, 1));
        assert(fern::size(result, 0) == fern::size(condition, 0));
        assert(fern::size(result, 1) == fern::size(condition, 1));

        if_then_2d_2d(input_no_data_policy, output_no_data_policy,
            IndexRanges<2>{
                IndexRange(0, fern::size(condition, 0)),
                IndexRange(0, fern::size(condition, 1)),
            }, condition, true_value, result);
    }

};


template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Condition,
    class TrueValue,
    class Result>
struct IfThenByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Condition,
    TrueValue,
    Result,
    ParallelExecutionPolicy,
    array_2d_tag,
    array_2d_tag>
{

    // if(2d, 2d, 2d)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ParallelExecutionPolicy const& /* execution_policy */,
        Condition const& condition,
        TrueValue const& true_value,
        Result& result)
    {
        assert(fern::size(true_value, 0) == fern::size(condition, 0));
        assert(fern::size(true_value, 1) == fern::size(condition, 1));
        assert(fern::size(result, 0) == fern::size(condition, 0));
        assert(fern::size(result, 1) == fern::size(condition, 1));

        ThreadPool& pool(ThreadClient::pool());
        size_t const size1 = fern::size(condition, 0);
        size_t const size2 = fern::size(condition, 1);
        std::vector<IndexRanges<2>> ranges = index_ranges(pool.size(),
            size1, size2);
        std::vector<std::future<void>> futures;
        futures.reserve(ranges.size());

        for(auto const& block_range: ranges) {
            auto function = std::bind(
                if_then_2d_2d<
                    InputNoDataPolicy, OutputNoDataPolicy,
                    Condition, TrueValue, Result>,
                std::cref(input_no_data_policy),
                std::ref(output_no_data_policy), std::cref(block_range),
                std::cref(condition), std::cref(true_value), std::ref(result));
            futures.emplace_back(pool.submit(function));
        }

        for(auto& future: futures) {
            future.get();
        }
    }

};


template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Condition,
    class TrueValue,
    class FalseValue,
    class Result>
struct IfThenElseByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Condition,
    TrueValue,
    FalseValue,
    Result,
    SequentialExecutionPolicy,
    array_2d_tag,
    array_2d_tag,
    array_2d_tag>
{

    // if(2d, 2d, 2d, 2d)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SequentialExecutionPolicy const& /* execution_policy */,
        Condition const& condition,
        TrueValue const& true_value,
        FalseValue const& false_value,
        Result& result)
    {
        assert(fern::size(true_value, 0) == fern::size(condition, 0));
        assert(fern::size(true_value, 1) == fern::size(condition, 1));
        assert(fern::size(false_value, 0) == fern::size(condition, 0));
        assert(fern::size(false_value, 1) == fern::size(condition, 1));
        assert(fern::size(result, 0) == fern::size(condition, 0));
        assert(fern::size(result, 1) == fern::size(condition, 1));

        if_then_else_2d_2d_2d(input_no_data_policy, output_no_data_policy,
            IndexRanges<2>{
                IndexRange(0, fern::size(condition, 0)),
                IndexRange(0, fern::size(condition, 1)),
            }, condition, true_value, false_value, result);
    }

};


template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Condition,
    class TrueValue,
    class FalseValue,
    class Result>
struct IfThenElseByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Condition,
    TrueValue,
    FalseValue,
    Result,
    ParallelExecutionPolicy,
    array_2d_tag,
    array_2d_tag,
    array_2d_tag>
{

    // if(2d, 2d, 2d, 2d)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ParallelExecutionPolicy const& /* execution_policy */,
        Condition const& condition,
        TrueValue const& true_value,
        FalseValue const& false_value,
        Result& result)
    {
        assert(fern::size(true_value, 0) == fern::size(condition, 0));
        assert(fern::size(true_value, 1) == fern::size(condition, 1));
        assert(fern::size(false_value, 0) == fern::size(condition, 0));
        assert(fern::size(false_value, 1) == fern::size(condition, 1));
        assert(fern::size(result, 0) == fern::size(condition, 0));
        assert(fern::size(result, 1) == fern::size(condition, 1));

        ThreadPool& pool(ThreadClient::pool());
        size_t const size1 = fern::size(condition, 0);
        size_t const size2 = fern::size(condition, 1);
        std::vector<IndexRanges<2>> ranges = index_ranges(pool.size(),
            size1, size2);
        std::vector<std::future<void>> futures;
        futures.reserve(ranges.size());

        for(auto const& block_range: ranges) {
            auto function = std::bind(
                if_then_else_2d_2d_2d<
                    InputNoDataPolicy, OutputNoDataPolicy,
                    Condition, TrueValue, FalseValue, Result>,
                std::cref(input_no_data_policy),
                std::ref(output_no_data_policy), std::cref(block_range),
                std::cref(condition), std::cref(true_value),
                std::cref(false_value), std::ref(result));
            futures.emplace_back(pool.submit(function));
        }

        for(auto& future: futures) {
            future.get();
        }
    }

};


template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Condition,
    class TrueValue,
    class Result,
    class ExecutionPolicy>
class IfThenByExecutionPolicy
{
};


template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Condition,
    class TrueValue,
    class Result>
struct IfThenByExecutionPolicy<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Condition,
    TrueValue,
    Result,
    ExecutionPolicy>
{

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ExecutionPolicy const& execution_policy,
        Condition const& condition,
        TrueValue const& true_value,
        Result& result)
    {
        switch(execution_policy.which()) {
            case fern::algorithm::detail::sequential_execution_policy_id: {
                IfThenByArgumentCategory<
                    InputNoDataPolicy,
                    OutputNoDataPolicy,
                    Condition,
                    TrueValue,
                    Result,
                    SequentialExecutionPolicy,
                    base_class<argument_category<Condition>, array_2d_tag>,
                    base_class<argument_category<TrueValue>, array_2d_tag>>
                        ::apply(
                            input_no_data_policy, output_no_data_policy,
                            fern::algorithm::detail::get_policy<
                                SequentialExecutionPolicy>(execution_policy),
                            condition, true_value, result);
                break;
            }
            case fern::algorithm::detail::parallel_execution_policy_id: {
                IfThenByArgumentCategory<
                    InputNoDataPolicy,
                    OutputNoDataPolicy,
                    Condition,
                    TrueValue,
                    Result,
                    ParallelExecutionPolicy,
                    base_class<argument_category<Condition>, array_2d_tag>,
                    base_class<argument_category<TrueValue>, array_2d_tag>>
                        ::apply(
                            input_no_data_policy, output_no_data_policy,
                            fern::algorithm::detail::get_policy<
                                ParallelExecutionPolicy>(execution_policy),
                            condition, true_value, result);
                break;
            }
        }
    }

};


template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Condition,
    class TrueValue,
    class FalseValue,
    class Result,
    class ExecutionPolicy>
class IfThenElseByExecutionPolicy
{
};


template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Condition,
    class TrueValue,
    class FalseValue,
    class Result>
struct IfThenElseByExecutionPolicy<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Condition,
    TrueValue,
    FalseValue,
    Result,
    ExecutionPolicy>
{

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ExecutionPolicy const& execution_policy,
        Condition const& condition,
        TrueValue const& true_value,
        FalseValue const& false_value,
        Result& result)
    {
        switch(execution_policy.which()) {
            case fern::algorithm::detail::sequential_execution_policy_id: {
                IfThenElseByArgumentCategory<
                    InputNoDataPolicy,
                    OutputNoDataPolicy,
                    Condition,
                    TrueValue,
                    FalseValue,
                    Result,
                    SequentialExecutionPolicy,
                    base_class<argument_category<Condition>, array_2d_tag>,
                    base_class<argument_category<TrueValue>, array_2d_tag>,
                    base_class<argument_category<FalseValue>, array_2d_tag>>
                        ::apply(
                        input_no_data_policy, output_no_data_policy,
                        fern::algorithm::detail::get_policy<
                            SequentialExecutionPolicy>(execution_policy),
                        condition, true_value, false_value, result);
                break;
            }
            case fern::algorithm::detail::parallel_execution_policy_id: {
                IfThenElseByArgumentCategory<
                    InputNoDataPolicy,
                    OutputNoDataPolicy,
                    Condition,
                    TrueValue,
                    FalseValue,
                    Result,
                    ParallelExecutionPolicy,
                    base_class<argument_category<Condition>, array_2d_tag>,
                    base_class<argument_category<TrueValue>, array_2d_tag>,
                    base_class<argument_category<FalseValue>, array_2d_tag>>
                        ::apply(
                            input_no_data_policy, output_no_data_policy,
                            fern::algorithm::detail::get_policy<
                                ParallelExecutionPolicy>(execution_policy),
                            condition, true_value, false_value, result);
                break;
            }
        }
    }

};

} // namespace dispatch


template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Condition,
    class TrueValue,
    class Result>
void if_(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy const& execution_policy,
    Condition const& condition,
    TrueValue const& true_value,
    Result& result)
{
    dispatch::IfThenByExecutionPolicy<InputNoDataPolicy, OutputNoDataPolicy,
        Condition, TrueValue, Result, ExecutionPolicy>::apply(
            input_no_data_policy, output_no_data_policy, execution_policy,
            condition, true_value, result);
}


template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Condition,
    class TrueValue,
    class FalseValue,
    class Result>
void if_(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy const& execution_policy,
    Condition const& condition,
    TrueValue const& true_value,
    FalseValue const& false_value,
    Result& result)
{
    dispatch::IfThenElseByExecutionPolicy<InputNoDataPolicy, OutputNoDataPolicy,
        Condition, TrueValue, FalseValue, Result, ExecutionPolicy>::apply(
            input_no_data_policy, output_no_data_policy, execution_policy,
            condition, true_value, false_value, result);
}

} // namespace detail
} // namespace if_
} // namespace algorithm
} // namespace fern
