// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <boost/algorithm/clamp.hpp>
#include "fern/core/base_class.h"
#include "fern/core/data_type_traits.h"
#include "fern/algorithm/core/index_ranges.h"
#include "fern/algorithm/policy/execution_policy.h"


namespace fern {
namespace algorithm {
namespace clamp {
namespace detail {

template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename LowerBound,
    typename UpperBound,
    typename Result>
static void clamp_0d_0d_0d(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    Value const& value,
    LowerBound const& lower_bound,
    UpperBound const& upper_bound,
    Result& result)
{
    assert(!std::get<1>(input_no_data_policy).is_no_data());
    assert(!std::get<2>(input_no_data_policy).is_no_data());

    if(std::get<0>(input_no_data_policy).is_no_data()) {
        output_no_data_policy.mark_as_no_data();
    }
    else {
        get(result) = boost::algorithm::clamp(get(value), get(lower_bound),
            get(upper_bound));
    }
}


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename LowerBound,
    typename UpperBound,
    typename Result>
static void clamp_1d_1d_1d(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    IndexRanges<1> const& index_ranges,
    Value const& value,
    LowerBound const& lower_bound,
    UpperBound const& upper_bound,
    Result& result)
{
    for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end(); ++i) {
        if(std::get<0>(input_no_data_policy).is_no_data(i)) {
            output_no_data_policy.mark_as_no_data(i);
        }
        else {
            assert(!std::get<1>(input_no_data_policy).is_no_data(i));
            assert(!std::get<2>(input_no_data_policy).is_no_data(i));

            get(result, i) = boost::algorithm::clamp(get(value, i),
                get(lower_bound, i), get(upper_bound, i));

        }
    }
}


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename LowerBound,
    typename UpperBound,
    typename Result>
static void clamp_1d_1d_0d(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    IndexRanges<1> const& index_ranges,
    Value const& value,
    LowerBound const& lower_bound,
    UpperBound const& upper_bound,
    Result& result)
{
    for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end(); ++i) {
        if(std::get<0>(input_no_data_policy).is_no_data(i)) {
            output_no_data_policy.mark_as_no_data(i);
        }
        else {
            assert(!std::get<1>(input_no_data_policy).is_no_data(i));
            assert(!std::get<2>(input_no_data_policy).is_no_data());

            get(result, i) = boost::algorithm::clamp(get(value, i),
                get(lower_bound, i), get(upper_bound));

        }
    }
}


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename LowerBound,
    typename UpperBound,
    typename Result>
static void clamp_1d_0d_1d(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    IndexRanges<1> const& index_ranges,
    Value const& value,
    LowerBound const& lower_bound,
    UpperBound const& upper_bound,
    Result& result)
{
    for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end(); ++i) {
        if(std::get<0>(input_no_data_policy).is_no_data(i)) {
            output_no_data_policy.mark_as_no_data(i);
        }
        else {
            assert(!std::get<1>(input_no_data_policy).is_no_data());
            assert(!std::get<2>(input_no_data_policy).is_no_data(i));

            get(result, i) = boost::algorithm::clamp(get(value, i),
                get(lower_bound), get(upper_bound, i));

        }
    }
}


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename LowerBound,
    typename UpperBound,
    typename Result>
static void clamp_2d_2d_2d(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    IndexRanges<2> const& index_ranges,
    Value const& value,
    LowerBound const& lower_bound,
    UpperBound const& upper_bound,
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
                assert(!std::get<1>(input_no_data_policy).is_no_data(index_));
                assert(!std::get<2>(input_no_data_policy).is_no_data(index_));

                get(result, index_) = boost::algorithm::clamp(
                    get(value, index_), get(lower_bound, index_),
                    get(upper_bound, index_));
            }

            ++index_;
        }
    }
}


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename LowerBound,
    typename UpperBound,
    typename Result>
static void clamp_2d_2d_0d(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    IndexRanges<2> const& index_ranges,
    Value const& value,
    LowerBound const& lower_bound,
    UpperBound const& upper_bound,
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
                assert(!std::get<1>(input_no_data_policy).is_no_data(index_));
                assert(!std::get<2>(input_no_data_policy).is_no_data());

                get(result, index_) = boost::algorithm::clamp(
                    get(value, index_), get(lower_bound, index_),
                    get(upper_bound));
            }

            ++index_;
        }
    }
}


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename LowerBound,
    typename UpperBound,
    typename Result>
static void clamp_2d_0d_2d(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    IndexRanges<2> const& index_ranges,
    Value const& value,
    LowerBound const& lower_bound,
    UpperBound const& upper_bound,
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
                assert(!std::get<1>(input_no_data_policy).is_no_data());
                assert(!std::get<2>(input_no_data_policy).is_no_data(index_));

                get(result, index_) = boost::algorithm::clamp(
                    get(value, index_), get(lower_bound),
                    get(upper_bound, index_));
            }

            ++index_;
        }
    }
}


namespace dispatch {

template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename LowerBound,
    typename UpperBound,
    typename Result,
    typename ExecutionPolicy,
    typename ValueCollectionCategory,
    typename UpperBoundCollectionCategory,
    typename LowerBoundCollectionCategory>
struct ClampByArgumentCategory
{
};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename LowerBound,
    typename UpperBound,
    typename Result,
    typename ExecutionPolicy>
struct ClampByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value,
    LowerBound,
    UpperBound,
    Result,
    ExecutionPolicy,
    array_0d_tag,
    array_0d_tag,
    array_0d_tag>
{

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ExecutionPolicy& /* execution_policy */,
        Value const& value,
        LowerBound const& lower_bound,
        UpperBound const& upper_bound,
        Result& result)
    {
        clamp_0d_0d_0d(input_no_data_policy, output_no_data_policy,
            value, lower_bound, upper_bound, result);
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename LowerBound,
    typename UpperBound,
    typename Result>
struct ClampByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value,
    LowerBound,
    UpperBound,
    Result,
    SequentialExecutionPolicy,
    array_1d_tag,
    array_1d_tag,
    array_1d_tag>
{

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SequentialExecutionPolicy& /* execution_policy */,
        Value const& value,
        LowerBound const& lower_bound,
        UpperBound const& upper_bound,
        Result& result)
    {
        assert(size(result) == size(value));
        assert(size(lower_bound) == size(value));
        assert(size(upper_bound) == size(value));

        clamp_1d_1d_1d(input_no_data_policy, output_no_data_policy,
            IndexRanges<1>{IndexRange(0, size(result))},
            value, lower_bound, upper_bound, result);
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename LowerBound,
    typename UpperBound,
    typename Result>
struct ClampByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value,
    LowerBound,
    UpperBound,
    Result,
    ParallelExecutionPolicy,
    array_1d_tag,
    array_1d_tag,
    array_1d_tag>
{

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ParallelExecutionPolicy& execution_policy,
        Value const& value,
        LowerBound const& lower_bound,
        UpperBound const& upper_bound,
        Result& result)
    {
        assert(size(result) == size(value));
        assert(size(lower_bound) == size(value));
        assert(size(upper_bound) == size(value));

        ThreadPool& pool(execution_policy.thread_pool());
        size_t const size_ = size(result);
        std::vector<IndexRanges<1>> ranges = index_ranges(pool.size(), size_);
        std::vector<std::future<void>> futures;
        futures.reserve(ranges.size());

        for(auto const& block_range: ranges) {
            auto function = std::bind(
                clamp_1d_1d_1d<
                    InputNoDataPolicy, OutputNoDataPolicy,
                    Value, UpperBound, LowerBound, Result>,
                std::cref(input_no_data_policy),
                std::ref(output_no_data_policy), std::cref(block_range),
                std::cref(value), std::cref(lower_bound),
                std::cref(upper_bound), std::ref(result));
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
    typename Value,
    typename LowerBound,
    typename UpperBound,
    typename Result>
struct ClampByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value,
    LowerBound,
    UpperBound,
    Result,
    SequentialExecutionPolicy,
    array_1d_tag,
    array_1d_tag,
    array_0d_tag>
{

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SequentialExecutionPolicy& /* execution_policy */,
        Value const& value,
        LowerBound const& lower_bound,
        UpperBound const& upper_bound,
        Result& result)
    {
        assert(size(result) == size(value));
        assert(size(lower_bound) == size(value));

        clamp_1d_1d_0d(input_no_data_policy, output_no_data_policy,
            IndexRanges<1>{IndexRange(0, size(result))},
            value, lower_bound, upper_bound, result);
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename LowerBound,
    typename UpperBound,
    typename Result>
struct ClampByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value,
    LowerBound,
    UpperBound,
    Result,
    ParallelExecutionPolicy,
    array_1d_tag,
    array_1d_tag,
    array_0d_tag>
{

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ParallelExecutionPolicy& execution_policy,
        Value const& value,
        LowerBound const& lower_bound,
        UpperBound const& upper_bound,
        Result& result)
    {
        assert(size(result) == size(value));
        assert(size(lower_bound) == size(value));

        ThreadPool& pool(execution_policy.thread_pool());
        size_t const size_ = size(result);
        std::vector<IndexRanges<1>> ranges = index_ranges(pool.size(), size_);
        std::vector<std::future<void>> futures;
        futures.reserve(ranges.size());

        for(auto const& block_range: ranges) {
            auto function = std::bind(
                clamp_1d_1d_1d<
                    InputNoDataPolicy, OutputNoDataPolicy,
                    Value, UpperBound, LowerBound, Result>,
                std::cref(input_no_data_policy),
                std::ref(output_no_data_policy), std::cref(block_range),
                std::cref(value), std::cref(lower_bound),
                std::cref(upper_bound), std::ref(result));
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
    typename Value,
    typename LowerBound,
    typename UpperBound,
    typename Result>
struct ClampByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value,
    LowerBound,
    UpperBound,
    Result,
    SequentialExecutionPolicy,
    array_1d_tag,
    array_0d_tag,
    array_1d_tag>
{

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SequentialExecutionPolicy& /* execution_policy */,
        Value const& value,
        LowerBound const& lower_bound,
        UpperBound const& upper_bound,
        Result& result)
    {
        assert(size(result) == size(value));
        assert(size(upper_bound) == size(value));

        clamp_1d_0d_1d(input_no_data_policy, output_no_data_policy,
            IndexRanges<1>{IndexRange(0, size(result))},
            value, lower_bound, upper_bound, result);
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename LowerBound,
    typename UpperBound,
    typename Result>
struct ClampByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value,
    LowerBound,
    UpperBound,
    Result,
    ParallelExecutionPolicy,
    array_1d_tag,
    array_0d_tag,
    array_1d_tag>
{

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ParallelExecutionPolicy& execution_policy,
        Value const& value,
        LowerBound const& lower_bound,
        UpperBound const& upper_bound,
        Result& result)
    {
        assert(size(result) == size(value));
        assert(size(upper_bound) == size(value));

        ThreadPool& pool(execution_policy.thread_pool());
        size_t const size_ = size(result);
        std::vector<IndexRanges<1>> ranges = index_ranges(pool.size(), size_);
        std::vector<std::future<void>> futures;
        futures.reserve(ranges.size());

        for(auto const& block_range: ranges) {
            auto function = std::bind(
                clamp_1d_1d_1d<
                    InputNoDataPolicy, OutputNoDataPolicy,
                    Value, UpperBound, LowerBound, Result>,
                std::cref(input_no_data_policy),
                std::ref(output_no_data_policy), std::cref(block_range),
                std::cref(value), std::cref(lower_bound),
                std::cref(upper_bound), std::ref(result));
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
    typename Value,
    typename LowerBound,
    typename UpperBound,
    typename Result>
struct ClampByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value,
    LowerBound,
    UpperBound,
    Result,
    SequentialExecutionPolicy,
    array_2d_tag,
    array_2d_tag,
    array_2d_tag>
{

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SequentialExecutionPolicy& /* execution_policy */,
        Value const& value,
        LowerBound const& lower_bound,
        UpperBound const& upper_bound,
        Result& result)
    {
        assert(size(result, 0) == size(value, 0));
        assert(size(result, 1) == size(value, 1));
        assert(size(lower_bound, 0) == size(value, 0));
        assert(size(lower_bound, 1) == size(value, 1));
        assert(size(upper_bound, 0) == size(value, 0));
        assert(size(upper_bound, 1) == size(value, 1));

        clamp_2d_2d_2d(input_no_data_policy, output_no_data_policy,
            IndexRanges<2>{
                IndexRange(0, size(result, 0)),
                IndexRange(0, size(result, 1))},
            value, lower_bound, upper_bound, result);
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename LowerBound,
    typename UpperBound,
    typename Result>
struct ClampByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value,
    LowerBound,
    UpperBound,
    Result,
    ParallelExecutionPolicy,
    array_2d_tag,
    array_2d_tag,
    array_2d_tag>
{

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ParallelExecutionPolicy& execution_policy,
        Value const& value,
        LowerBound const& lower_bound,
        UpperBound const& upper_bound,
        Result& result)
    {
        assert(size(result, 0) == size(value, 0));
        assert(size(result, 1) == size(value, 1));
        assert(size(lower_bound, 0) == size(value, 0));
        assert(size(lower_bound, 1) == size(value, 1));
        assert(size(upper_bound, 0) == size(value, 0));
        assert(size(upper_bound, 1) == size(value, 1));

        ThreadPool& pool(execution_policy.thread_pool());
        size_t const size1 = size(result, 0);
        size_t const size2 = size(result, 1);
        std::vector<IndexRanges<2>> ranges = index_ranges(pool.size(),
            size1, size2);
        std::vector<std::future<void>> futures;
        futures.reserve(ranges.size());

        for(auto const& block_range: ranges) {
            auto function = std::bind(
                clamp_2d_2d_2d<
                    InputNoDataPolicy, OutputNoDataPolicy,
                    Value, UpperBound, LowerBound, Result>,
                std::cref(input_no_data_policy),
                std::ref(output_no_data_policy), std::cref(block_range),
                std::cref(value), std::cref(lower_bound),
                std::cref(upper_bound), std::ref(result));
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
    typename Value,
    typename LowerBound,
    typename UpperBound,
    typename Result>
struct ClampByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value,
    LowerBound,
    UpperBound,
    Result,
    SequentialExecutionPolicy,
    array_2d_tag,
    array_2d_tag,
    array_0d_tag>
{

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SequentialExecutionPolicy& /* execution_policy */,
        Value const& value,
        LowerBound const& lower_bound,
        UpperBound const& upper_bound,
        Result& result)
    {
        assert(size(result, 0) == size(value, 0));
        assert(size(result, 1) == size(value, 1));
        assert(size(lower_bound, 0) == size(value, 0));
        assert(size(lower_bound, 1) == size(value, 1));

        clamp_2d_2d_0d(input_no_data_policy, output_no_data_policy,
            IndexRanges<2>{
                IndexRange(0, size(result, 0)),
                IndexRange(0, size(result, 1))},
            value, lower_bound, upper_bound, result);
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename LowerBound,
    typename UpperBound,
    typename Result>
struct ClampByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value,
    LowerBound,
    UpperBound,
    Result,
    ParallelExecutionPolicy,
    array_2d_tag,
    array_2d_tag,
    array_0d_tag>
{

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ParallelExecutionPolicy& execution_policy,
        Value const& value,
        LowerBound const& lower_bound,
        UpperBound const& upper_bound,
        Result& result)
    {
        assert(size(result, 0) == size(value, 0));
        assert(size(result, 1) == size(value, 1));
        assert(size(lower_bound, 0) == size(value, 0));
        assert(size(lower_bound, 1) == size(value, 1));

        ThreadPool& pool(execution_policy.thread_pool());
        size_t const size1 = size(result, 0);
        size_t const size2 = size(result, 1);
        std::vector<IndexRanges<2>> ranges = index_ranges(pool.size(),
            size1, size2);
        std::vector<std::future<void>> futures;
        futures.reserve(ranges.size());

        for(auto const& block_range: ranges) {
            auto function = std::bind(
                clamp_2d_2d_0d<
                    InputNoDataPolicy, OutputNoDataPolicy,
                    Value, UpperBound, LowerBound, Result>,
                std::cref(input_no_data_policy),
                std::ref(output_no_data_policy), std::cref(block_range),
                std::cref(value), std::cref(lower_bound),
                std::cref(upper_bound), std::ref(result));
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
    typename Value,
    typename LowerBound,
    typename UpperBound,
    typename Result>
struct ClampByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value,
    LowerBound,
    UpperBound,
    Result,
    SequentialExecutionPolicy,
    array_2d_tag,
    array_0d_tag,
    array_2d_tag>
{

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SequentialExecutionPolicy& /* execution_policy */,
        Value const& value,
        LowerBound const& lower_bound,
        UpperBound const& upper_bound,
        Result& result)
    {
        assert(size(result, 0) == size(value, 0));
        assert(size(result, 1) == size(value, 1));
        assert(size(upper_bound, 0) == size(value, 0));
        assert(size(upper_bound, 1) == size(value, 1));

        clamp_2d_0d_2d(input_no_data_policy, output_no_data_policy,
            IndexRanges<2>{
                IndexRange(0, size(result, 0)),
                IndexRange(0, size(result, 1))},
            value, lower_bound, upper_bound, result);
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename LowerBound,
    typename UpperBound,
    typename Result>
struct ClampByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value,
    LowerBound,
    UpperBound,
    Result,
    ParallelExecutionPolicy,
    array_2d_tag,
    array_0d_tag,
    array_2d_tag>
{

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ParallelExecutionPolicy& execution_policy,
        Value const& value,
        LowerBound const& lower_bound,
        UpperBound const& upper_bound,
        Result& result)
    {
        assert(size(result, 0) == size(value, 0));
        assert(size(result, 1) == size(value, 1));
        assert(size(upper_bound, 0) == size(value, 0));
        assert(size(upper_bound, 1) == size(value, 1));

        ThreadPool& pool(execution_policy.thread_pool());
        size_t const size1 = size(result, 0);
        size_t const size2 = size(result, 1);
        std::vector<IndexRanges<2>> ranges = index_ranges(pool.size(),
            size1, size2);
        std::vector<std::future<void>> futures;
        futures.reserve(ranges.size());

        for(auto const& block_range: ranges) {
            auto function = std::bind(
                clamp_2d_0d_2d<
                    InputNoDataPolicy, OutputNoDataPolicy,
                    Value, UpperBound, LowerBound, Result>,
                std::cref(input_no_data_policy),
                std::ref(output_no_data_policy), std::cref(block_range),
                std::cref(value), std::cref(lower_bound),
                std::cref(upper_bound), std::ref(result));
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
    typename Value,
    typename LowerBound,
    typename UpperBound,
    typename Result,
    typename ExecutionPolicy>
struct ClampByExecutionPolicy
{

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ExecutionPolicy& execution_policy,
        Value const& value,
        LowerBound const& lower_bound,
        UpperBound const& upper_bound,
        Result& result)
    {
        ClampByArgumentCategory<
            InputNoDataPolicy,
            OutputNoDataPolicy,
            Value,
            LowerBound,
            UpperBound,
            Result,
            ExecutionPolicy,
            base_class<argument_category<Value>, array_2d_tag>,
            base_class<argument_category<LowerBound>, array_2d_tag>,
            base_class<argument_category<UpperBound>, array_2d_tag>>::apply(
                input_no_data_policy, output_no_data_policy, execution_policy,
                value, lower_bound, upper_bound, result);
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename LowerBound,
    typename UpperBound,
    typename Result>
struct ClampByExecutionPolicy<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value,
    LowerBound,
    UpperBound,
    Result,
    ExecutionPolicy>
{

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ExecutionPolicy& execution_policy,
        Value const& value,
        LowerBound const& lower_bound,
        UpperBound const& upper_bound,
        Result& result)
    {
        switch(execution_policy.which()) {
            case fern::algorithm::detail::sequential_execution_policy_id: {
                ClampByArgumentCategory<
                    InputNoDataPolicy,
                    OutputNoDataPolicy,
                    Value,
                    LowerBound,
                    UpperBound,
                    Result,
                    SequentialExecutionPolicy,
                    base_class<argument_category<Value>, array_2d_tag>,
                    base_class<argument_category<LowerBound>, array_2d_tag>,
                    base_class<argument_category<UpperBound>, array_2d_tag>>
                        ::apply(
                            input_no_data_policy, output_no_data_policy,
                            boost::get<SequentialExecutionPolicy>(
                                execution_policy),
                            value, lower_bound, upper_bound, result);
                break;
            }
            case fern::algorithm::detail::parallel_execution_policy_id: {
                ClampByArgumentCategory<
                    InputNoDataPolicy,
                    OutputNoDataPolicy,
                    Value,
                    LowerBound,
                    UpperBound,
                    Result,
                    ParallelExecutionPolicy,
                    base_class<argument_category<Value>, array_2d_tag>,
                    base_class<argument_category<LowerBound>, array_2d_tag>,
                    base_class<argument_category<UpperBound>, array_2d_tag>>
                        ::apply(
                            input_no_data_policy, output_no_data_policy,
                            boost::get<ParallelExecutionPolicy>(
                                execution_policy),
                            value, lower_bound, upper_bound, result);
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
    typename Value,
    typename LowerBound,
    typename UpperBound,
    typename Result>
static void clamp(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy& execution_policy,
    Value const& value,
    LowerBound const& lower_bound,
    UpperBound const& upper_bound,
    Result& result)
{
    dispatch::ClampByExecutionPolicy<InputNoDataPolicy, OutputNoDataPolicy,
        Value, LowerBound, UpperBound, Result, ExecutionPolicy>::apply(
            input_no_data_policy, output_no_data_policy, execution_policy,
            value, lower_bound, upper_bound, result);
}

} // namespace detail
} // namespace clamp
} // namespace algorithm
} // namespace fern
