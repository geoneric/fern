#pragma once
#include "fern/core/base_class.h"
#include "fern/core/type_traits.h"
#include "fern/algorithm/policy/execution_policy.h"


namespace fern {
namespace offset {
namespace detail {

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
    IndexRange const& index_range{index_ranges[0]};
    size_t const nr_elements{fern::size(value)};
    value_type<Offset_> offset(fern::get<0>(offset_));
    size_t first_element_to_copy, end_element_to_copy;

    // Make sure offset isn't larger than the number of elements, otherwise
    // calculations of indices below may go out of range.

    if(offset <= 0) {
        // Copy values from the back of the source range to the front of the
        // destination range.
        offset = std::max(offset, -static_cast<int>(nr_elements));
        first_element_to_copy = std::max(static_cast<size_t>(std::abs(offset)),
            index_range.begin());
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
        offset = std::min(offset, static_cast<value_type<Offset_>>(
            nr_elements));
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


    // Copy the values.
    for(size_t i = first_element_to_copy; i < end_element_to_copy; ++i) {
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
    size_t const nr_elements{fern::size(value)};
    value_type<Offset_> offset(fern::get<0>(offset_));

    // Make sure offset isn't larger than the number of elements, otherwise
    // calculations of indices below may go out of range.

    if(offset <= 0) {
        offset = std::max(offset, -static_cast<int>(nr_elements));

        for(size_t i = nr_elements + offset; i < nr_elements; ++i) {
            output_no_data_policy.mark_as_no_data(i);
        }
    }
    else {
        offset = std::min(offset, static_cast<value_type<Offset_>>(
            nr_elements));

        for(size_t i = 0; i < static_cast<size_t>(offset); ++i) {
            output_no_data_policy.mark_as_no_data(i);
        }
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
    size_t const nr_elements{fern::size(result)};
    value_type<Offset_> offset(fern::get<0>(offset_));

    // Make sure offset isn't larger than the number of elements, otherwise
    // calculations of indices below may go out of range.

    if(offset <= 0) {
        offset = std::max(offset, -static_cast<int>(nr_elements));

        for(size_t i = nr_elements + offset; i < nr_elements; ++i) {
            fern::get(result, i) = fill_value;
        }

    }
    else {
        offset = std::min(offset, static_cast<value_type<Offset_>>(
            nr_elements));

        for(size_t i = 0; i < static_cast<size_t>(offset); ++i) {
            fern::get(result, i) = fill_value;
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
            case fern::detail::sequential_execution_policy_id: {
                detail::dispatch::Offset<
                    InputNoDataPolicy,
                    OutputNoDataPolicy,
                    Value,
                    Offset_,
                    Result,
                    SequentialExecutionPolicy,
                    array_1d_tag>::apply(
                        input_no_data_policy, output_no_data_policy,
                        fern::detail::get_policy<SequentialExecutionPolicy>(
                            execution_policy),
                        value, offset_, result);
                break;
            }
            case fern::detail::parallel_execution_policy_id: {
                detail::dispatch::Offset<
                    InputNoDataPolicy,
                    OutputNoDataPolicy,
                    Value,
                    Offset_,
                    Result,
                    ParallelExecutionPolicy,
                    array_1d_tag>::apply(
                        input_no_data_policy, output_no_data_policy,
                        fern::detail::get_policy<ParallelExecutionPolicy>(
                            execution_policy),
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
            case fern::detail::sequential_execution_policy_id: {
                detail::dispatch::Offset<
                    InputNoDataPolicy,
                    OutputNoDataPolicy,
                    Value,
                    Offset_,
                    Result,
                    SequentialExecutionPolicy,
                    array_1d_tag>::apply(
                        input_no_data_policy, output_no_data_policy,
                        fern::detail::get_policy<SequentialExecutionPolicy>(
                            execution_policy),
                        value, offset_, fill_value, result);
                break;
            }
            case fern::detail::parallel_execution_policy_id: {
                detail::dispatch::Offset<
                    InputNoDataPolicy,
                    OutputNoDataPolicy,
                    Value,
                    Offset_,
                    Result,
                    ParallelExecutionPolicy,
                    array_1d_tag>::apply(
                        input_no_data_policy, output_no_data_policy,
                        fern::detail::get_policy<ParallelExecutionPolicy>(
                            execution_policy),
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
} // namespace fern
