#pragma once
#include "fern/core/array_2d_traits.h"
#include "fern/core/base_class.h"
#include "fern/core/constant_traits.h"
#include "fern/algorithm/core/index_ranges.h"
#include "fern/algorithm/policy/execution_policy.h"


namespace fern {
namespace algorithm {
namespace if_ {
namespace detail {

template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename Result>
void if_then_0d_0d(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    Condition const& condition,
    TrueValue const& true_value,
    Result& result)
{
    if(!std::get<0>(input_no_data_policy).is_no_data() && get(condition) &&
            !std::get<1>(input_no_data_policy).is_no_data()) {
        get(result) = get(true_value);
    }
    else {
        output_no_data_policy.mark_as_no_data();
    }
}


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename Result>
void if_then_2d_0d(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    IndexRanges<2> const& index_ranges,
    Condition const& condition,
    TrueValue const& true_value,
    Result& result)
{
    size_t index_;

    for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end(); ++i) {

        index_ = index(result, i, index_ranges[1].begin());

        for(size_t j = index_ranges[1].begin(); j < index_ranges[1].end();
                ++j) {

            if(!std::get<0>(input_no_data_policy).is_no_data(index_) &&
                    get(condition, index_) &&
                    !std::get<1>(input_no_data_policy).is_no_data(index_)) {
                get(result, index_) = get(true_value);
            }
            else {
                output_no_data_policy.mark_as_no_data(index_);
            }

            ++index_;
        }
    }
}


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename Result>
void if_then_2d_2d(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    IndexRanges<2> const& index_ranges,
    Condition const& condition,
    TrueValue const& true_value,
    Result& result)
{
    size_t index_;

    for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end(); ++i) {

        index_ = index(result, i, index_ranges[1].begin());

        for(size_t j = index_ranges[1].begin(); j < index_ranges[1].end();
                ++j) {

            if(!std::get<0>(input_no_data_policy).is_no_data(index_) &&
                    get(condition, index_) &&
                    !std::get<1>(input_no_data_policy).is_no_data(index_)) {
                get(result, index_) = get(true_value, index_);
            }
            else {
                output_no_data_policy.mark_as_no_data(index_);
            }

            ++index_;
        }
    }
}


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename FalseValue,
    typename Result>
void if_then_else_0d_0d_0d(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    Condition const& condition,
    TrueValue const& true_value,
    FalseValue const& false_value,
    Result& result)
{
    if(std::get<0>(input_no_data_policy).is_no_data()) {
        output_no_data_policy.mark_as_no_data();
    }
    else if(get(condition)) {
        if(std::get<1>(input_no_data_policy).is_no_data()) {
            output_no_data_policy.mark_as_no_data();
        }
        else {
            get(result) = get(true_value);
        }
    }
    else {
        if(std::get<2>(input_no_data_policy).is_no_data()) {
            output_no_data_policy.mark_as_no_data();
        }
        else {
            get(result) = get(false_value);
        }
    }
}


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename FalseValue,
    typename Result>
void if_then_else_2d_2d_2d(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    IndexRanges<2> const& index_ranges,
    Condition const& condition,
    TrueValue const& true_value,
    FalseValue const& false_value,
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
                if(get(condition, index_)) {
                    if(std::get<1>(input_no_data_policy).is_no_data(index_)) {
                        output_no_data_policy.mark_as_no_data(index_);
                    }
                    else {
                        get(result, index_) = get(true_value, index_);
                    }
                }
                else {
                    if(std::get<2>(input_no_data_policy).is_no_data(index_)) {
                        output_no_data_policy.mark_as_no_data(index_);
                    }
                    else {
                        get(result, index_) = get(false_value, index_);
                    }
                }
            }

            ++index_;
        }
    }
}


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename FalseValue,
    typename Result>
void if_then_else_2d_2d_0d(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    IndexRanges<2> const& index_ranges,
    Condition const& condition,
    TrueValue const& true_value,
    FalseValue const& false_value,
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
                if(get(condition, index_)) {
                    if(std::get<1>(input_no_data_policy).is_no_data(index_)) {
                        output_no_data_policy.mark_as_no_data(index_);
                    }
                    else {
                        get(result, index_) = get(true_value, index_);
                    }
                }
                else {
                    if(std::get<2>(input_no_data_policy).is_no_data(index_)) {
                        output_no_data_policy.mark_as_no_data(index_);
                    }
                    else {
                        get(result, index_) = get(false_value);
                    }
                }
            }

            ++index_;
        }
    }
}


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename FalseValue,
    typename Result>
void if_then_else_2d_0d_2d(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    IndexRanges<2> const& index_ranges,
    Condition const& condition,
    TrueValue const& true_value,
    FalseValue const& false_value,
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
                if(get(condition, index_)) {
                    if(std::get<1>(input_no_data_policy).is_no_data(index_)) {
                        output_no_data_policy.mark_as_no_data(index_);
                    }
                    else {
                        get(result, index_) = get(true_value);
                    }
                }
                else {
                    if(std::get<2>(input_no_data_policy).is_no_data(index_)) {
                        output_no_data_policy.mark_as_no_data(index_);
                    }
                    else {
                        get(result, index_) = get(false_value, index_);
                    }
                }
            }

            ++index_;
        }
    }
}


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename FalseValue,
    typename Result>
void if_then_else_2d_0d_0d(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    IndexRanges<2> const& index_ranges,
    Condition const& condition,
    TrueValue const& true_value,
    FalseValue const& false_value,
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
                if(get(condition, index_)) {
                    if(std::get<1>(input_no_data_policy).is_no_data(index_)) {
                        output_no_data_policy.mark_as_no_data(index_);
                    }
                    else {
                        get(result, index_) = get(true_value);
                    }
                }
                else {
                    if(std::get<2>(input_no_data_policy).is_no_data(index_)) {
                        output_no_data_policy.mark_as_no_data(index_);
                    }
                    else {
                        get(result, index_) = get(false_value);
                    }
                }
            }

            ++index_;
        }
    }
}


namespace dispatch {

template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename Result,
    typename ExecutionPolicy,
    typename ConditionCollectionCategory,
    typename TrueValueCollectionCategory>
class IfThenByArgumentCategory
{
};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename FalseValue,
    typename Result,
    typename ExecutionPolicy,
    typename ConditionCollectionCategory,
    typename TrueValueCollectionCategory,
    typename FalseValueCollectionCategory>
struct IfThenElseByArgumentCategory
{
};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename Result,
    typename ExecutionPolicy>
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
        ExecutionPolicy& /* execution_policy */,
        Condition const& condition,
        TrueValue const& true_value,
        Result& result)
    {
        if_then_0d_0d(input_no_data_policy, output_no_data_policy,
            condition, true_value, result);
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename FalseValue,
    typename Result,
    typename ExecutionPolicy>
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
        ExecutionPolicy& /* execution_policy */,
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
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename Result>
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
        SequentialExecutionPolicy& /* execution_policy */,
        Condition const& condition,
        TrueValue const& true_value,
        Result& result)
    {
        assert(size(result, 0) == size(condition, 0));
        assert(size(result, 1) == size(condition, 1));

        if_then_2d_0d(input_no_data_policy, output_no_data_policy,
            IndexRanges<2>{
                IndexRange(0, size(condition, 0)),
                IndexRange(0, size(condition, 1)),
            }, condition, true_value, result);
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename Result>
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
        ParallelExecutionPolicy& execution_policy,
        Condition const& condition,
        TrueValue const& true_value,
        Result& result)
    {
        assert(size(result, 0) == size(condition, 0));
        assert(size(result, 1) == size(condition, 1));

        ThreadPool& pool(execution_policy.thread_pool());
        size_t const size1 = size(condition, 0);
        size_t const size2 = size(condition, 1);
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
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename FalseValue,
    typename Result>
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
        SequentialExecutionPolicy& /* execution_policy */,
        Condition const& condition,
        TrueValue const& true_value,
        FalseValue const& false_value,
        Result& result)
    {
        assert(size(result, 0) == size(condition, 0));
        assert(size(result, 1) == size(condition, 1));

        if_then_else_2d_0d_0d(input_no_data_policy, output_no_data_policy,
            IndexRanges<2>{
                IndexRange(0, size(condition, 0)),
                IndexRange(0, size(condition, 1)),
            }, condition, true_value, false_value, result);
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename FalseValue,
    typename Result>
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
        ParallelExecutionPolicy& execution_policy,
        Condition const& condition,
        TrueValue const& true_value,
        FalseValue const& false_value,
        Result& result)
    {
        assert(size(result, 0) == size(condition, 0));
        assert(size(result, 1) == size(condition, 1));

        ThreadPool& pool(execution_policy.thread_pool());
        size_t const size1 = size(condition, 0);
        size_t const size2 = size(condition, 1);
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
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename Result>
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
        SequentialExecutionPolicy& /* execution_policy */,
        Condition const& condition,
        TrueValue const& true_value,
        Result& result)
    {
        assert(size(true_value, 0) == size(condition, 0));
        assert(size(true_value, 1) == size(condition, 1));
        assert(size(result, 0) == size(condition, 0));
        assert(size(result, 1) == size(condition, 1));

        if_then_2d_2d(input_no_data_policy, output_no_data_policy,
            IndexRanges<2>{
                IndexRange(0, size(condition, 0)),
                IndexRange(0, size(condition, 1)),
            }, condition, true_value, result);
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename Result>
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
        ParallelExecutionPolicy& execution_policy,
        Condition const& condition,
        TrueValue const& true_value,
        Result& result)
    {
        assert(size(true_value, 0) == size(condition, 0));
        assert(size(true_value, 1) == size(condition, 1));
        assert(size(result, 0) == size(condition, 0));
        assert(size(result, 1) == size(condition, 1));

        ThreadPool& pool(execution_policy.thread_pool());
        size_t const size1 = size(condition, 0);
        size_t const size2 = size(condition, 1);
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
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename FalseValue,
    typename Result>
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
        SequentialExecutionPolicy& /* execution_policy */,
        Condition const& condition,
        TrueValue const& true_value,
        FalseValue const& false_value,
        Result& result)
    {
        assert(size(true_value, 0) == size(condition, 0));
        assert(size(true_value, 1) == size(condition, 1));
        assert(size(false_value, 0) == size(condition, 0));
        assert(size(false_value, 1) == size(condition, 1));
        assert(size(result, 0) == size(condition, 0));
        assert(size(result, 1) == size(condition, 1));

        if_then_else_2d_2d_2d(input_no_data_policy, output_no_data_policy,
            IndexRanges<2>{
                IndexRange(0, size(condition, 0)),
                IndexRange(0, size(condition, 1)),
            }, condition, true_value, false_value, result);
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename FalseValue,
    typename Result>
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
        ParallelExecutionPolicy& execution_policy,
        Condition const& condition,
        TrueValue const& true_value,
        FalseValue const& false_value,
        Result& result)
    {
        assert(size(true_value, 0) == size(condition, 0));
        assert(size(true_value, 1) == size(condition, 1));
        assert(size(false_value, 0) == size(condition, 0));
        assert(size(false_value, 1) == size(condition, 1));
        assert(size(result, 0) == size(condition, 0));
        assert(size(result, 1) == size(condition, 1));

        ThreadPool& pool(execution_policy.thread_pool());
        size_t const size1 = size(condition, 0);
        size_t const size2 = size(condition, 1);
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
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename FalseValue,
    typename Result>
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
    array_0d_tag>
{

    // if(2d, 2d, 0d, 2d)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SequentialExecutionPolicy& /* execution_policy */,
        Condition const& condition,
        TrueValue const& true_value,
        FalseValue const& false_value,
        Result& result)
    {
        assert(size(true_value, 0) == size(condition, 0));
        assert(size(true_value, 1) == size(condition, 1));
        assert(size(result, 0) == size(condition, 0));
        assert(size(result, 1) == size(condition, 1));

        if_then_else_2d_2d_0d(input_no_data_policy, output_no_data_policy,
            IndexRanges<2>{
                IndexRange(0, size(condition, 0)),
                IndexRange(0, size(condition, 1)),
            }, condition, true_value, false_value, result);
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename FalseValue,
    typename Result>
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
    array_0d_tag>
{

    // if(2d, 2d, 0d, 2d)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ParallelExecutionPolicy& execution_policy,
        Condition const& condition,
        TrueValue const& true_value,
        FalseValue const& false_value,
        Result& result)
    {
        assert(size(true_value, 0) == size(condition, 0));
        assert(size(true_value, 1) == size(condition, 1));
        assert(size(result, 0) == size(condition, 0));
        assert(size(result, 1) == size(condition, 1));

        ThreadPool& pool(execution_policy.thread_pool());
        size_t const size1 = size(condition, 0);
        size_t const size2 = size(condition, 1);
        std::vector<IndexRanges<2>> ranges = index_ranges(pool.size(),
            size1, size2);
        std::vector<std::future<void>> futures;
        futures.reserve(ranges.size());

        for(auto const& block_range: ranges) {
            auto function = std::bind(
                if_then_else_2d_2d_0d<
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
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename FalseValue,
    typename Result>
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
    array_2d_tag>
{

    // if(2d, 0d, 2d, 2d)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SequentialExecutionPolicy& /* execution_policy */,
        Condition const& condition,
        TrueValue const& true_value,
        FalseValue const& false_value,
        Result& result)
    {
        assert(size(false_value, 0) == size(condition, 0));
        assert(size(false_value, 1) == size(condition, 1));
        assert(size(result, 0) == size(condition, 0));
        assert(size(result, 1) == size(condition, 1));

        if_then_else_2d_0d_2d(input_no_data_policy, output_no_data_policy,
            IndexRanges<2>{
                IndexRange(0, size(condition, 0)),
                IndexRange(0, size(condition, 1)),
            }, condition, true_value, false_value, result);
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename FalseValue,
    typename Result>
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
    array_2d_tag>
{

    // if(2d, 0d, 2d, 2d)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ParallelExecutionPolicy& execution_policy,
        Condition const& condition,
        TrueValue const& true_value,
        FalseValue const& false_value,
        Result& result)
    {
        assert(size(false_value, 0) == size(condition, 0));
        assert(size(false_value, 1) == size(condition, 1));
        assert(size(result, 0) == size(condition, 0));
        assert(size(result, 1) == size(condition, 1));

        ThreadPool& pool(execution_policy.thread_pool());
        size_t const size1 = size(condition, 0);
        size_t const size2 = size(condition, 1);
        std::vector<IndexRanges<2>> ranges = index_ranges(pool.size(),
            size1, size2);
        std::vector<std::future<void>> futures;
        futures.reserve(ranges.size());

        for(auto const& block_range: ranges) {
            auto function = std::bind(
                if_then_else_2d_0d_2d<
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
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename Result,
    typename ExecutionPolicy>
struct IfThenByExecutionPolicy
{

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ExecutionPolicy& execution_policy,
        Condition const& condition,
        TrueValue const& true_value,
        Result& result)
    {
        IfThenByArgumentCategory<InputNoDataPolicy, OutputNoDataPolicy,
            Condition, TrueValue, Result,
            ExecutionPolicy,
            base_class<argument_category<Condition>, array_2d_tag>,
            base_class<argument_category<TrueValue>, array_2d_tag>>
                ::apply(input_no_data_policy, output_no_data_policy,
                    execution_policy, condition, true_value, result);

    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename Result>
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
        ExecutionPolicy& execution_policy,
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
                            boost::get<SequentialExecutionPolicy>(
                                execution_policy),
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
                            boost::get<ParallelExecutionPolicy>(
                                execution_policy),
                            condition, true_value, result);
                break;
            }
        }
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename FalseValue,
    typename Result,
    typename ExecutionPolicy>
struct IfThenElseByExecutionPolicy
{

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ExecutionPolicy& execution_policy,
        Condition const& condition,
        TrueValue const& true_value,
        FalseValue const& false_value,
        Result& result)
    {
        IfThenElseByArgumentCategory<
            InputNoDataPolicy, OutputNoDataPolicy,
            Condition, TrueValue, FalseValue, Result,
            ExecutionPolicy,
            base_class<argument_category<Condition>, array_2d_tag>,
            base_class<argument_category<TrueValue>, array_2d_tag>,
            base_class<argument_category<FalseValue>, array_2d_tag>>
                ::apply(input_no_data_policy, output_no_data_policy,
                    execution_policy, condition, true_value, false_value,
                    result);
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Condition,
    typename TrueValue,
    typename FalseValue,
    typename Result>
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
        ExecutionPolicy& execution_policy,
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
                            boost::get<SequentialExecutionPolicy>(
                                execution_policy),
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
                            boost::get<ParallelExecutionPolicy>(
                                execution_policy),
                            condition, true_value, false_value, result);
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
    typename Condition,
    typename TrueValue,
    typename Result>
void if_(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy& execution_policy,
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
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename ExecutionPolicy,
    typename Condition,
    typename TrueValue,
    typename FalseValue,
    typename Result>
void if_(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy& execution_policy,
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
