#pragma once
#include "fern/core/argument_traits.h"
#include "fern/core/base_class.h"
#include "fern/core/collection_traits.h"
#include "fern/core/type_traits.h"
#include "fern/algorithm/core/index_ranges.h"
#include "fern/algorithm/policy/execution_policy.h"


namespace fern {
namespace algorithm {
namespace offset {
namespace detail {

template<
    typename Offset>
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
    typename Offset>
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
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Offset_,
    typename Result>
void copy_1d(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    IndexRanges<1> const& index_ranges,
    Value const& value,
    Offset_ const& offset_,
    Result& result)
{
    value_type<Offset_> offset(get<0>(offset_));

    IndexRange range_to_copy_(range_to_copy(index_ranges[0], size(value),
        offset));

    // Copy the values.
    for(size_t i = range_to_copy_.begin(); i < range_to_copy_.end(); ++i) {

        if(std::get<0>(input_no_data_policy).is_no_data(i)) {
            output_no_data_policy.mark_as_no_data(i + offset);
        }
        else {
            get(result, i + offset) = get(value, i);
        }
    }
}


template<
    typename OutputNoDataPolicy,
    typename Value,
    typename Offset_>
void mark_no_data_1d(
    OutputNoDataPolicy& output_no_data_policy,
    Value const& value,
    Offset_ const& offset_)
{
    IndexRange const range_to_initialize_(range_to_initialize(
        size(value), get<0>(offset_)));

    for(size_t i = range_to_initialize_.begin();
            i < range_to_initialize_.end(); ++i) {
        output_no_data_policy.mark_as_no_data(i);
    }
}


template<
    typename Offset_,
    typename Result>
void fill_value_1d(
    Offset_ const& offset_,
    value_type<Result> const& fill_value,
    Result& result)
{
    IndexRange const range_to_initialize_(range_to_initialize(
        size(result), get<0>(offset_)));

    for(size_t i = range_to_initialize_.begin();
            i < range_to_initialize_.end(); ++i) {
        get(result, i) = fill_value;
    }
}


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Offset_,
    typename Result>
void copy_2d(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    IndexRanges<2> const& index_ranges,
    Value const& value,
    Offset_ const& offset_,
    Result& result)
{
    value_type<Offset_> offset1(get<0>(offset_));
    value_type<Offset_> offset2(get<1>(offset_));

    IndexRange const range_to_copy1(range_to_copy(index_ranges[0],
        size(value, 0), offset1));
    IndexRange const range_to_copy2(range_to_copy(index_ranges[1],
        size(value, 1), offset2));

    size_t source_index, destination_index;

    // Copy the values.
    for(size_t i = range_to_copy1.begin(); i < range_to_copy1.end(); ++i) {

        source_index = index(value, i, range_to_copy2.begin());
        destination_index = index(result, i + offset1, range_to_copy2.begin() +
                offset2);

        for(size_t j = range_to_copy2.begin(); j < range_to_copy2.end(); ++j) {

            if(std::get<0>(input_no_data_policy).is_no_data(source_index)) {
                output_no_data_policy.mark_as_no_data(destination_index);
            }
            else {
                get(result, destination_index) = get(value, source_index);
            }

            ++source_index;
            ++destination_index;
        }
    }
}


template<
    typename OutputNoDataPolicy,
    typename Value,
    typename Offset_>
void mark_no_data_2d(
    OutputNoDataPolicy& output_no_data_policy,
    Value const& value,
    Offset_ const& offset_)
{
    size_t const size1{size(value, 0)};
    size_t const size2{size(value, 1)};

    IndexRange const range_to_initialize1(range_to_initialize(size1,
        get<0>(offset_)));
    IndexRange const range_to_initialize2(range_to_initialize(size2,
        get<1>(offset_)));

    size_t index_;

    for(size_t i = range_to_initialize1.begin();
            i < range_to_initialize1.end(); ++i) {

        index_ = index(value, i, 0);

        for(size_t j = 0; j < size2; ++j) {
            output_no_data_policy.mark_as_no_data(index_);
            ++index_;
        }
    }

    for(size_t i = 0; i < size1; ++i) {

        index_ = index(value, i, range_to_initialize2.begin());

        for(size_t j = range_to_initialize2.begin();
                j < range_to_initialize2.end(); ++j) {
            output_no_data_policy.mark_as_no_data(index_);
            ++index_;
        }
    }
}


template<
    typename Offset_,
    typename Result>
void fill_value_2d(
    Offset_ const& offset_,
    value_type<Result> const& fill_value,
    Result& result)
{
    size_t const size1{size(result, 0)};
    size_t const size2{size(result, 1)};

    IndexRange const range_to_initialize1(range_to_initialize(size1,
        get<0>(offset_)));
    IndexRange const range_to_initialize2(range_to_initialize(size2,
        get<1>(offset_)));

    size_t index_;

    for(size_t i = range_to_initialize1.begin();
            i < range_to_initialize1.end(); ++i) {

        index_ = index(result, i, 0);

        for(size_t j = 0; j < size2; ++j) {
            get(result, index_) = fill_value;
            ++index_;
        }
    }

    for(size_t i = 0; i < size1; ++i) {

        index_ = index(result, i, range_to_initialize2.begin());

        for(size_t j = range_to_initialize2.begin();
                j < range_to_initialize2.end(); ++j) {
            get(result, index_) = fill_value;
            ++index_;
        }
    }
}


namespace dispatch {

template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Offset_,
    typename Result,
    typename ExecutionPolicy,
    typename ValueCollectionCategory>
class Offset
{
};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Offset_,
    typename Result>
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
        SequentialExecutionPolicy& /* execution_policy */,
        Value const& value,
        Offset_ const& offset_,
        Result& result)
    {
        assert(size(value) > 0);

        copy_1d(input_no_data_policy, output_no_data_policy,
            IndexRanges<1>{IndexRange(0, size(value))},
            value, offset_, result);
        mark_no_data_1d(output_no_data_policy, value, offset_);
    }

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SequentialExecutionPolicy& /* execution_policy */,
        Value const& value,
        Offset_ const& offset_,
        value_type<Result> const& fill_value,
        Result& result)
    {
        assert(size(value) > 0);

        copy_1d(input_no_data_policy, output_no_data_policy,
            IndexRanges<1>{IndexRange(0, size(value))},
            value, offset_, result);
        fill_value_1d(offset_, fill_value, result);
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Offset_,
    typename Result>
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
        ParallelExecutionPolicy& execution_policy,
        Value const& value,
        Offset_ const& offset_,
        Result& result)
    {
        assert(size(value) > 0);

        ThreadPool& pool(execution_policy.thread_pool());
        size_t const size_ = size(value);
        std::vector<IndexRanges<1>> ranges = index_ranges(pool.size(), size_);
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
        ParallelExecutionPolicy& execution_policy,
        Value const& value,
        Offset_ const& offset_,
        value_type<Result> const& fill_value,
        Result& result)
    {
        assert(size(value) > 0);

        ThreadPool& pool(execution_policy.thread_pool());
        size_t const size_ = size(value);
        std::vector<IndexRanges<1>> ranges = index_ranges(pool.size(), size_);
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
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Offset_,
    typename Result>
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
        ExecutionPolicy& execution_policy,
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
                        boost::get<SequentialExecutionPolicy>(execution_policy),
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
                        boost::get<ParallelExecutionPolicy>(execution_policy),
                        value, offset_, result);
                break;
            }
        }
    }

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ExecutionPolicy& execution_policy,
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
                        boost::get<SequentialExecutionPolicy>(execution_policy),
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
                        boost::get<ParallelExecutionPolicy>(execution_policy),
                        value, offset_, fill_value, result);
                break;
            }
        }
    }
};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Offset_,
    typename Result>
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
        SequentialExecutionPolicy& /* execution_policy */,
        Value const& value,
        Offset_ const& offset_,
        Result& result)
    {
        assert(size(value) > 0);

        copy_2d(input_no_data_policy, output_no_data_policy,
            IndexRanges<2>{
                IndexRange(0, size(value, 0)),
                IndexRange(0, size(value, 1)),
            }, value, offset_, result);
        mark_no_data_2d(output_no_data_policy, value, offset_);
    }

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SequentialExecutionPolicy& /* execution_policy */,
        Value const& value,
        Offset_ const& offset_,
        value_type<Result> const& fill_value,
        Result& result)
    {
        assert(size(value) > 0);

        copy_2d(input_no_data_policy, output_no_data_policy,
            IndexRanges<2>{
                IndexRange(0, size(value, 0)),
                IndexRange(0, size(value, 1)),
            }, value, offset_, result);
        fill_value_2d(offset_, fill_value, result);
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Offset_,
    typename Result>
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
        ParallelExecutionPolicy& execution_policy,
        Value const& value,
        Offset_ const& offset_,
        Result& result)
    {
        assert(size(value) > 0);

        ThreadPool& pool(execution_policy.thread_pool());
        size_t const size1 = size(value, 0);
        size_t const size2 = size(value, 1);
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
        ParallelExecutionPolicy& execution_policy,
        Value const& value,
        Offset_ const& offset_,
        value_type<Result> const& fill_value,
        Result& result)
    {
        assert(size(value) > 0);

        ThreadPool& pool(execution_policy.thread_pool());
        size_t const size1 = size(value, 0);
        size_t const size2 = size(value, 1);
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
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Offset_,
    typename Result>
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
        ExecutionPolicy& execution_policy,
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
                        boost::get<SequentialExecutionPolicy>(execution_policy),
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
                        boost::get<ParallelExecutionPolicy>(execution_policy),
                        value, offset_, result);
                break;
            }
        }
    }

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ExecutionPolicy& execution_policy,
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
                        boost::get<SequentialExecutionPolicy>(execution_policy),
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
                        boost::get<ParallelExecutionPolicy>(execution_policy),
                        value, offset_, fill_value, result);
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
    typename Offset,
    typename Result
>
void offset(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy& execution_policy,
    Value const& value,
    Offset const& offset,
    Result& result)
{
    if(size(value) == 0) {
        return;
    }

    dispatch::Offset<InputNoDataPolicy, OutputNoDataPolicy,
        Value, Offset, Result, ExecutionPolicy,
        base_class<argument_category<Value>, array_2d_tag>>::apply(
            input_no_data_policy, output_no_data_policy, execution_policy,
            value, offset, result);
}


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename ExecutionPolicy,
    typename Value,
    typename Offset,
    typename Result
>
void offset(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy& execution_policy,
    Value const& value,
    Offset const& offset,
    value_type<Result> const& fill_value,
    Result& result)
{
    if(size(value) == 0) {
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
