#pragma once
#include "fern/core/argument_traits.h"
#include "fern/core/base_class.h"
#include "fern/core/collection_traits.h"
#include "fern/core/thread_client.h"
#include "fern/core/type_traits.h"
#include "fern/algorithm/core/index_ranges.h"
#include "fern/algorithm/policy/execution_policy.h"


namespace fern {
namespace algorithm {
namespace offset {
namespace detail {

template<
    class Offset>
IndexRange range_to_copy(
    IndexRange const& index_range,
    size_t const nr_elements,
    Offset& offset)  // offset is updated.
{
    size_t first_element_to_copy, end_element_to_copy;

    // Make sure offset isn't larger than the number of elements, otherwise
    // calculations of indices may go out of range.

    if(offset <= 0) {

        // Copy values from the back of the source range to the front of the
        // destination range.
        offset = std::max(offset, -static_cast<Offset>(nr_elements));
        first_element_to_copy = std::max(
            static_cast<size_t>(std::abs(offset)), index_range.begin());
        end_element_to_copy = index_range.end();

        // Clamp index to the index range passed in.
        first_element_to_copy = std::min(first_element_to_copy,
            end_element_to_copy);

        assert(first_element_to_copy >= index_range.begin());
        assert(first_element_to_copy <= index_range.end());
        assert(end_element_to_copy >= index_range.begin());
        assert(end_element_to_copy <= index_range.end());
    }
    else {

        // Copy values from the front of the source range to the back of the
        // destination range.
        offset = std::min(offset, static_cast<Offset>(nr_elements));
        first_element_to_copy = index_range.begin();
        end_element_to_copy = std::min(index_range.end(),
            nr_elements - static_cast<size_t>(std::abs(offset)));

        // Clamp index to the index range passed in.
        end_element_to_copy = std::max(first_element_to_copy,
            end_element_to_copy);

        assert(first_element_to_copy >= index_range.begin());
        assert(first_element_to_copy < index_range.end());
        assert(end_element_to_copy >= index_range.begin());
        assert(end_element_to_copy <= index_range.end());
    }

    return IndexRange(first_element_to_copy, end_element_to_copy);
}


template<
    class Offset>
IndexRange range_to_initialize(
    size_t const nr_elements,
    Offset offset)
{
    size_t first_element_to_initialize, end_element_to_initialize;

    // Make sure offset isn't larger than the number of elements, otherwise
    // calculations of indices below may go out of range.

    if(offset <= 0) {
        offset = std::max(offset, -static_cast<Offset>(nr_elements));
        first_element_to_initialize = nr_elements + offset;
        end_element_to_initialize = nr_elements;
    }
    else {
        offset = std::min(offset, static_cast<Offset>(nr_elements));
        first_element_to_initialize = 0;
        end_element_to_initialize = static_cast<size_t>(offset);
    }

    return IndexRange(first_element_to_initialize, end_element_to_initialize);
}


template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value,
    class Offset_,
    class Result>
void copy_1d(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    IndexRanges<1> const& index_ranges,
    Value const& value,
    Offset_ const& offset_,
    Result& result)
{
    value_type<Offset_> offset(fern::get<0>(offset_));

    IndexRange range_to_copy_(range_to_copy(index_ranges[0], fern::size(value),
        offset));

    // Copy the values.
    for(size_t i = range_to_copy_.begin(); i < range_to_copy_.end(); ++i) {

        if(input_no_data_policy.is_no_data(i)) {
            output_no_data_policy.mark_as_no_data(i + offset);
        }
        else {
            fern::get(result, i + offset) = fern::get(value, i);
        }
    }
}


template<
    class OutputNoDataPolicy,
    class Value,
    class Offset_>
void mark_no_data_1d(
    OutputNoDataPolicy& output_no_data_policy,
    Value const& value,
    Offset_ const& offset_)
{
    IndexRange const range_to_initialize_(range_to_initialize(
        fern::size(value), fern::get<0>(offset_)));

    for(size_t i = range_to_initialize_.begin();
            i < range_to_initialize_.end(); ++i) {
        output_no_data_policy.mark_as_no_data(i);
    }
}


template<
    class Offset_,
    class Result>
void fill_value_1d(
    Offset_ const& offset_,
    value_type<Result> const& fill_value,
    Result& result)
{
    IndexRange const range_to_initialize_(range_to_initialize(
        fern::size(result), fern::get<0>(offset_)));

    for(size_t i = range_to_initialize_.begin();
            i < range_to_initialize_.end(); ++i) {
        fern::get(result, i) = fill_value;
    }
}


template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value,
    class Offset_,
    class Result>
void copy_2d(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    IndexRanges<2> const& index_ranges,
    Value const& value,
    Offset_ const& offset_,
    Result& result)
{
    value_type<Offset_> offset1(fern::get<0>(offset_));
    value_type<Offset_> offset2(fern::get<1>(offset_));

    IndexRange const range_to_copy1(range_to_copy(index_ranges[0],
        fern::size(value, 0), offset1));
    IndexRange const range_to_copy2(range_to_copy(index_ranges[1],
        fern::size(value, 1), offset2));

    // Copy the values.
    for(size_t i = range_to_copy1.begin(); i < range_to_copy1.end(); ++i) {
        for(size_t j = range_to_copy2.begin(); j < range_to_copy2.end(); ++j) {

            if(input_no_data_policy.is_no_data(i, j)) {
                output_no_data_policy.mark_as_no_data(i + offset1, j + offset2);
            }
            else {
                fern::get(result, i + offset1, j + offset2) = fern::get(value,
                    i, j);
            }
        }
    }
}


template<
    class OutputNoDataPolicy,
    class Value,
    class Offset_>
void mark_no_data_2d(
    OutputNoDataPolicy& output_no_data_policy,
    Value const& value,
    Offset_ const& offset_)
{
    size_t const size1{fern::size(value, 0)};
    size_t const size2{fern::size(value, 1)};

    IndexRange const range_to_initialize1(range_to_initialize(size1,
        fern::get<0>(offset_)));
    IndexRange const range_to_initialize2(range_to_initialize(size2,
        fern::get<1>(offset_)));

    for(size_t i = range_to_initialize1.begin();
            i < range_to_initialize1.end(); ++i) {
        for(size_t j = 0; j < size2; ++j) {
            output_no_data_policy.mark_as_no_data(i, j);
        }
    }

    for(size_t i = 0; i < size1; ++i) {
        for(size_t j = range_to_initialize2.begin();
                j < range_to_initialize2.end(); ++j) {
            output_no_data_policy.mark_as_no_data(i, j);
        }
    }
}


template<
    class Offset_,
    class Result>
void fill_value_2d(
    Offset_ const& offset_,
    value_type<Result> const& fill_value,
    Result& result)
{
    size_t const size1{fern::size(result, 0)};
    size_t const size2{fern::size(result, 1)};

    IndexRange const range_to_initialize1(range_to_initialize(size1,
        fern::get<0>(offset_)));
    IndexRange const range_to_initialize2(range_to_initialize(size2,
        fern::get<1>(offset_)));

    for(size_t i = range_to_initialize1.begin();
            i < range_to_initialize1.end(); ++i) {
        for(size_t j = 0; j < size2; ++j) {
            fern::get(result, i, j) = fill_value;
        }
    }

    for(size_t i = 0; i < size1; ++i) {
        for(size_t j = range_to_initialize2.begin();
                j < range_to_initialize2.end(); ++j) {
            fern::get(result, i, j) = fill_value;
        }
    }
}


namespace dispatch {

template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value,
    class Offset_,
    class Result,
    class ExecutionPolicy,
    class ValueCollectionCategory>
class Offset
{
};


template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value,
    class Offset_,
    class Result>
struct Offset<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value,
    Offset_,
    Result,
    SequentialExecutionPolicy,
    array_1d_tag>
{

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SequentialExecutionPolicy const& /* execution_policy */,
        Value const& value,
        Offset_ const& offset_,
        Result& result)
    {
        assert(fern::size(value) > 0);

        copy_1d(input_no_data_policy, output_no_data_policy,
            IndexRanges<1>{IndexRange(0, fern::size(value))},
            value, offset_, result);
        mark_no_data_1d(output_no_data_policy, value, offset_);
    }

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SequentialExecutionPolicy const& /* execution_policy */,
        Value const& value,
        Offset_ const& offset_,
        value_type<Result> const& fill_value,
        Result& result)
    {
        assert(fern::size(value) > 0);

        copy_1d(input_no_data_policy, output_no_data_policy,
            IndexRanges<1>{IndexRange(0, fern::size(value))},
            value, offset_, result);
        fill_value_1d(offset_, fill_value, result);
    }

};


template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value,
    class Offset_,
    class Result>
struct Offset<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value,
    Offset_,
    Result,
    ParallelExecutionPolicy,
    array_1d_tag>
{

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ParallelExecutionPolicy const& /* execution_policy */,
        Value const& value,
        Offset_ const& offset_,
        Result& result)
    {
        assert(fern::size(value) > 0);

        ThreadPool& pool(ThreadClient::pool());
        size_t const size = fern::size(value);
        std::vector<IndexRanges<1>> ranges = index_ranges(pool.size(), size);
        std::vector<std::future<void>> futures;
        futures.reserve(ranges.size());

        // Handle elements to copy.
        for(auto const& block_range: ranges) {
            auto function = std::bind(
                copy_1d<
                    InputNoDataPolicy, OutputNoDataPolicy,
                    Value, Offset_, Result>,
                std::cref(input_no_data_policy),
                std::ref(output_no_data_policy), std::cref(block_range),
                std::cref(value), std::cref(offset_), std::ref(result));
            futures.emplace_back(pool.submit(function));
        }

        // Handle elements to mark as no-data.
        auto function = std::bind(
            mark_no_data_1d<OutputNoDataPolicy, Value, Offset_>,
                std::ref(output_no_data_policy), std::cref(value),
                std::cref(offset_));
        futures.emplace_back(pool.submit(function));

        for(auto& future: futures) {
            future.get();
        }
    }

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ParallelExecutionPolicy const& /* execution_policy */,
        Value const& value,
        Offset_ const& offset_,
        value_type<Result> const& fill_value,
        Result& result)
    {
        assert(fern::size(value) > 0);

        ThreadPool& pool(ThreadClient::pool());
        size_t const size = fern::size(value);
        std::vector<IndexRanges<1>> ranges = index_ranges(pool.size(), size);
        std::vector<std::future<void>> futures;
        futures.reserve(ranges.size());

        // Handle elements to copy.
        for(auto const& block_range: ranges) {
            auto function = std::bind(
                copy_1d<
                    InputNoDataPolicy, OutputNoDataPolicy,
                    Value, Offset_, Result>,
                std::cref(input_no_data_policy),
                std::ref(output_no_data_policy), std::cref(block_range),
                std::cref(value), std::cref(offset_), std::ref(result));
            futures.emplace_back(pool.submit(function));
        }

        // Handle elements to mark as no-data.
        auto function = std::bind(
            fill_value_1d<Offset_, Result>,
                std::cref(offset_), std::cref(fill_value), std::ref(result));
        futures.emplace_back(pool.submit(function));

        for(auto& future: futures) {
            future.get();
        }
    }
};


template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value,
    class Offset_,
    class Result>
struct Offset<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value,
    Offset_,
    Result,
    ExecutionPolicy,
    array_1d_tag>
{

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ExecutionPolicy const& execution_policy,
        Value const& value,
        Offset_ const& offset_,
        Result& result)
    {
        switch(execution_policy.which()) {
            case fern::algorithm::detail::sequential_execution_policy_id: {
                detail::dispatch::Offset<
                    InputNoDataPolicy,
                    OutputNoDataPolicy,
                    Value,
                    Offset_,
                    Result,
                    SequentialExecutionPolicy,
                    array_1d_tag>::apply(
                        input_no_data_policy, output_no_data_policy,
                        fern::algorithm::detail::get_policy<
                            SequentialExecutionPolicy>(execution_policy),
                        value, offset_, result);
                break;
            }
            case fern::algorithm::detail::parallel_execution_policy_id: {
                detail::dispatch::Offset<
                    InputNoDataPolicy,
                    OutputNoDataPolicy,
                    Value,
                    Offset_,
                    Result,
                    ParallelExecutionPolicy,
                    array_1d_tag>::apply(
                        input_no_data_policy, output_no_data_policy,
                        fern::algorithm::detail::get_policy<
                            ParallelExecutionPolicy>(execution_policy),
                        value, offset_, result);
                break;
            }
        }
    }

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ExecutionPolicy const& execution_policy,
        Value const& value,
        Offset_ const& offset_,
        value_type<Result> const& fill_value,
        Result& result)
    {
        switch(execution_policy.which()) {
            case fern::algorithm::detail::sequential_execution_policy_id: {
                detail::dispatch::Offset<
                    InputNoDataPolicy,
                    OutputNoDataPolicy,
                    Value,
                    Offset_,
                    Result,
                    SequentialExecutionPolicy,
                    array_1d_tag>::apply(
                        input_no_data_policy, output_no_data_policy,
                        fern::algorithm::detail::get_policy<
                            SequentialExecutionPolicy>(execution_policy),
                        value, offset_, fill_value, result);
                break;
            }
            case fern::algorithm::detail::parallel_execution_policy_id: {
                detail::dispatch::Offset<
                    InputNoDataPolicy,
                    OutputNoDataPolicy,
                    Value,
                    Offset_,
                    Result,
                    ParallelExecutionPolicy,
                    array_1d_tag>::apply(
                        input_no_data_policy, output_no_data_policy,
                        fern::algorithm::detail::get_policy<
                            ParallelExecutionPolicy>(execution_policy),
                        value, offset_, fill_value, result);
                break;
            }
        }
    }
};


template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value,
    class Offset_,
    class Result>
struct Offset<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value,
    Offset_,
    Result,
    SequentialExecutionPolicy,
    array_2d_tag>
{

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SequentialExecutionPolicy const& /* execution_policy */,
        Value const& value,
        Offset_ const& offset_,
        Result& result)
    {
        assert(fern::size(value) > 0);

        copy_2d(input_no_data_policy, output_no_data_policy,
            IndexRanges<2>{
                IndexRange(0, fern::size(value, 0)),
                IndexRange(0, fern::size(value, 1)),
            }, value, offset_, result);
        mark_no_data_2d(output_no_data_policy, value, offset_);
    }

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SequentialExecutionPolicy const& /* execution_policy */,
        Value const& value,
        Offset_ const& offset_,
        value_type<Result> const& fill_value,
        Result& result)
    {
        assert(fern::size(value) > 0);

        copy_2d(input_no_data_policy, output_no_data_policy,
            IndexRanges<2>{
                IndexRange(0, fern::size(value, 0)),
                IndexRange(0, fern::size(value, 1)),
            }, value, offset_, result);
        fill_value_2d(offset_, fill_value, result);
    }

};


template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value,
    class Offset_,
    class Result>
struct Offset<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value,
    Offset_,
    Result,
    ParallelExecutionPolicy,
    array_2d_tag>
{

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ParallelExecutionPolicy const& /* execution_policy */,
        Value const& value,
        Offset_ const& offset_,
        Result& result)
    {
        assert(fern::size(value) > 0);

        ThreadPool& pool(ThreadClient::pool());
        size_t const size1 = fern::size(value, 0);
        size_t const size2 = fern::size(value, 1);
        std::vector<IndexRanges<2>> ranges = index_ranges(pool.size(),
            size1, size2);
        std::vector<std::future<void>> futures;
        futures.reserve(ranges.size());

        // Handle elements to copy.
        for(auto const& block_range: ranges) {
            auto function = std::bind(
                copy_2d<
                    InputNoDataPolicy, OutputNoDataPolicy,
                    Value, Offset_, Result>,
                std::cref(input_no_data_policy),
                std::ref(output_no_data_policy), std::cref(block_range),
                std::cref(value), std::cref(offset_), std::ref(result));
            futures.emplace_back(pool.submit(function));
        }

        // Handle elements to mark as no-data.
        auto function = std::bind(
            mark_no_data_2d<OutputNoDataPolicy, Value, Offset_>,
                std::ref(output_no_data_policy), std::cref(value),
                std::cref(offset_));
        futures.emplace_back(pool.submit(function));

        for(auto& future: futures) {
            future.get();
        }
    }

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ParallelExecutionPolicy const& /* execution_policy */,
        Value const& value,
        Offset_ const& offset_,
        value_type<Result> const& fill_value,
        Result& result)
    {
        assert(fern::size(value) > 0);

        ThreadPool& pool(ThreadClient::pool());
        size_t const size1 = fern::size(value, 0);
        size_t const size2 = fern::size(value, 1);
        std::vector<IndexRanges<2>> ranges = index_ranges(pool.size(),
            size1, size2);
        std::vector<std::future<void>> futures;
        futures.reserve(ranges.size());

        // Handle elements to copy.
        for(auto const& block_range: ranges) {
            auto function = std::bind(
                copy_2d<
                    InputNoDataPolicy, OutputNoDataPolicy,
                    Value, Offset_, Result>,
                std::cref(input_no_data_policy),
                std::ref(output_no_data_policy), std::cref(block_range),
                std::cref(value), std::cref(offset_), std::ref(result));
            futures.emplace_back(pool.submit(function));
        }

        // Handle elements to mark as no-data.
        auto function = std::bind(
            fill_value_2d<Offset_, Result>,
                std::cref(offset_), std::cref(fill_value), std::ref(result));
        futures.emplace_back(pool.submit(function));

        for(auto& future: futures) {
            future.get();
        }
    }
};


template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value,
    class Offset_,
    class Result>
struct Offset<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value,
    Offset_,
    Result,
    ExecutionPolicy,
    array_2d_tag>
{

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ExecutionPolicy const& execution_policy,
        Value const& value,
        Offset_ const& offset_,
        Result& result)
    {
        switch(execution_policy.which()) {
            case fern::algorithm::detail::sequential_execution_policy_id: {
                detail::dispatch::Offset<
                    InputNoDataPolicy,
                    OutputNoDataPolicy,
                    Value,
                    Offset_,
                    Result,
                    SequentialExecutionPolicy,
                    array_2d_tag>::apply(
                        input_no_data_policy, output_no_data_policy,
                        fern::algorithm::detail::get_policy<
                            SequentialExecutionPolicy>(execution_policy),
                        value, offset_, result);
                break;
            }
            case fern::algorithm::detail::parallel_execution_policy_id: {
                detail::dispatch::Offset<
                    InputNoDataPolicy,
                    OutputNoDataPolicy,
                    Value,
                    Offset_,
                    Result,
                    ParallelExecutionPolicy,
                    array_2d_tag>::apply(
                        input_no_data_policy, output_no_data_policy,
                        fern::algorithm::detail::get_policy<
                            ParallelExecutionPolicy>(execution_policy),
                        value, offset_, result);
                break;
            }
        }
    }

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ExecutionPolicy const& execution_policy,
        Value const& value,
        Offset_ const& offset_,
        value_type<Result> const& fill_value,
        Result& result)
    {
        switch(execution_policy.which()) {
            case fern::algorithm::detail::sequential_execution_policy_id: {
                detail::dispatch::Offset<
                    InputNoDataPolicy,
                    OutputNoDataPolicy,
                    Value,
                    Offset_,
                    Result,
                    SequentialExecutionPolicy,
                    array_2d_tag>::apply(
                        input_no_data_policy, output_no_data_policy,
                        fern::algorithm::detail::get_policy<
                            SequentialExecutionPolicy>(execution_policy),
                        value, offset_, fill_value, result);
                break;
            }
            case fern::algorithm::detail::parallel_execution_policy_id: {
                detail::dispatch::Offset<
                    InputNoDataPolicy,
                    OutputNoDataPolicy,
                    Value,
                    Offset_,
                    Result,
                    ParallelExecutionPolicy,
                    array_2d_tag>::apply(
                        input_no_data_policy, output_no_data_policy,
                        fern::algorithm::detail::get_policy<
                            ParallelExecutionPolicy>(execution_policy),
                        value, offset_, fill_value, result);
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
    class Value,
    class Offset,
    class Result
>
void offset(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Offset const& offset,
    Result& result)
{
    if(fern::size(value) == 0) {
        return;
    }

    dispatch::Offset<InputNoDataPolicy, OutputNoDataPolicy,
        Value, Offset, Result, ExecutionPolicy,
        base_class<argument_category<Value>, array_2d_tag>>::apply(
            input_no_data_policy, output_no_data_policy, execution_policy,
            value, offset, result);
}


template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Value,
    class Offset,
    class Result
>
void offset(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Offset const& offset,
    value_type<Result> const& fill_value,
    Result& result)
{
    if(fern::size(value) == 0) {
        return;
    }

    dispatch::Offset<InputNoDataPolicy, OutputNoDataPolicy,
        Value, Offset, Result, ExecutionPolicy,
        base_class<argument_category<Value>, array_2d_tag>>::apply(
            input_no_data_policy, output_no_data_policy, execution_policy,
            value, offset, fill_value, result);
}

} // namespace detail
} // namespace offset
} // namespace algorithm
} // namespace fern
