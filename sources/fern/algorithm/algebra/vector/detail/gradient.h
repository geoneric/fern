#pragma once
#include <algorithm>
#include "fern/core/base_class.h"
#include "fern/core/thread_client.h"
#include "fern/algorithm/core/index_ranges.h"


namespace fern {
namespace gradient {
namespace detail {

template<
    class Value,
    class Distance>
inline constexpr Value gradient(
    Value const& value1,
    Value const& value2,
    Distance const& distance)
{
    return (value2 - value1) / distance;
}


template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value,
    class Result>
static void gradient_x_2d(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& /* output_no_data_policy */,
    IndexRanges<2> const& index_ranges,
    Value const& value,
    Result& result)
{
    size_t const size2{fern::size(value, 1)};
    double const distance1{fern::cell_size(value, 0)};
    double const distance2{distance1 + distance1};

    // Handle left border, in case it is within the index range.
    for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end(); ++i) {
        for(size_t j = index_ranges[1].begin(); j < 1; ++j) {

            // Left cell of window lies outside of the raster. Consider it
            // no-data.

            if(!input_no_data_policy.is_no_data(i, j)) {

                assert(j + 1 < size2);
                if(!input_no_data_policy.is_no_data(i, j + 1)) {
                    // x c r
                    fern::get(result, i, j) = gradient(
                        fern::get(value, i, j),
                        fern::get(value, i, j + 1),
                        distance1);
                }
                else  {
                    // x c x
                    fern::get(result, i, j) = 0;
                }
            }
        }
    }


    // Handle right border, in case it is within the index range.
    for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end(); ++i) {
        for(size_t j = size2 - 1; j < index_ranges[1].end(); ++j) {

            // Right cell of window lies outside of the raster. Consider it
            // no-data.

            if(!input_no_data_policy.is_no_data(i, j)) {

                assert(j > 0);
                if(!input_no_data_policy.is_no_data(i, j - 1)) {
                    // l c x
                    fern::get(result, i, j) = gradient(
                        fern::get(value, i, j - 1),
                        fern::get(value, i, j),
                        distance1);
                }
                else  {
                    // x c x
                    fern::get(result, i, j) = 0;
                }
            }
        }
    }


    // Handle innert part, except for the borders (in case borders are within
    // the index range).
    for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end(); ++i) {
        for(size_t j = std::max<size_t>(1u, index_ranges[1].begin());
                j < std::min<size_t>(size2 - 1u, index_ranges[1].end()); ++j) {

            // All cells of window lie within the raster.

            if(!input_no_data_policy.is_no_data(i, j)) {

                assert(j > 0);
                assert(j + 1 < size2);

                if(!input_no_data_policy.is_no_data(i, j - 1)) {
                    if(!input_no_data_policy.is_no_data(i, j + 1)) {
                        // l c r
                        fern::get(result, i, j) = gradient(
                            fern::get(value, i, j - 1),
                            fern::get(value, i, j + 1),
                            distance2);
                    }
                    else {
                        // l c x
                        fern::get(result, i, j) = gradient(
                            fern::get(value, i, j - 1),
                            fern::get(value, i, j),
                            distance1);
                    }
                }
                else if(!input_no_data_policy.is_no_data(i, j + 1)) {
                    // x c r
                    fern::get(result, i, j) = gradient(
                        fern::get(value, i, j),
                        fern::get(value, i, j + 1),
                        distance1);
                }
                else {
                    // x c x
                    fern::get(result, i, j) = 0;
                }
            }
        }
    }
}


template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value,
    class Result>
static void gradient_y_2d(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& /* output_no_data_policy */,
    IndexRanges<2> const& index_ranges,
    Value const& value,
    Result& result)
{
    size_t const size1{fern::size(value, 0)};
    double const distance1{fern::cell_size(value, 1)};
    double const distance2{distance1 + distance1};

    // Handle top border, in case it is within the index range.
    for(size_t i = index_ranges[0].begin(); i < 1; ++i) {
        for(size_t j = index_ranges[1].begin(); j < index_ranges[1].end();
            ++j) {

            // Top cell of window lies outside of the raster. Consider it
            // no-data.

            if(!input_no_data_policy.is_no_data(i, j)) {

                assert(i + 1 < size1);
                if(!input_no_data_policy.is_no_data(i + 1, j)) {
                    // x c r
                    fern::get(result, i, j) = gradient(
                        fern::get(value, i, j),
                        fern::get(value, i + 1, j),
                        distance1);
                }
                else  {
                    // x c x
                    fern::get(result, i, j) = 0;
                }
            }
        }
    }


    // Handle bottom border, in case it is within the index range.
    for(size_t i = size1 - 1; i < index_ranges[0].end(); ++i) {
        for(size_t j = index_ranges[1].begin(); j < index_ranges[1].end();
                ++j) {

            // Bottom cell of window lies outside of the raster. Consider it
            // no-data.

            if(!input_no_data_policy.is_no_data(i, j)) {

                assert(i > 0);
                if(!input_no_data_policy.is_no_data(i - 1, j)) {
                    // l c x
                    fern::get(result, i, j) = gradient(
                        fern::get(value, i - 1, j),
                        fern::get(value, i, j),
                        distance1);
                }
                else  {
                    // x c x
                    fern::get(result, i, j) = 0;
                }
            }
        }
    }


    // Handle innert part, except for the borders (in case borders are within
    // the index range).
    for(size_t i = std::max<size_t>(1u, index_ranges[0].begin());
            i < std::min<size_t>(size1 - 1u, index_ranges[0].end()); ++i) {
        for(size_t j = index_ranges[1].begin(); j < index_ranges[1].end();
                ++j) {

            // All cells of window lie within the raster.

            if(!input_no_data_policy.is_no_data(i, j)) {

                assert(i > 0);
                assert(i + 1 < size1);

                if(!input_no_data_policy.is_no_data(i - 1, j)) {
                    if(!input_no_data_policy.is_no_data(i + 1, j)) {
                        // l c r
                        fern::get(result, i, j) = gradient(
                            fern::get(value, i - 1, j),
                            fern::get(value, i + 1, j),
                            distance2);
                    }
                    else {
                        // l c x
                        fern::get(result, i, j) = gradient(
                            fern::get(value, i - 1, j),
                            fern::get(value, i, j),
                            distance1);
                    }
                }
                else if(!input_no_data_policy.is_no_data(i + 1, j)) {
                    // x c r
                    fern::get(result, i, j) = gradient(
                        fern::get(value, i, j),
                        fern::get(value, i + 1, j),
                        distance1);
                }
                else {
                    // x c x
                    fern::get(result, i, j) = 0;
                }
            }
        }
    }
}


namespace dispatch {

template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value,
    class Result,
    class ExecutionPolicy,
    class ValueCollectionCategory>
struct GradientXByArgumentCategory
{
};


template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value,
    class Result>
struct GradientXByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value,
    Result,
    SequentialExecutionPolicy,
    raster_2d_tag>
{

    // gradient(2d)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SequentialExecutionPolicy const& /* execution_policy */,
        Value const& value,
        Result& result)
    {
        assert(fern::size(value, 0) == fern::size(result, 0));
        assert(fern::size(value, 1) == fern::size(result, 1));

        gradient_x_2d(input_no_data_policy, output_no_data_policy,
            IndexRanges<2>{
                IndexRange(0, fern::size(result, 0)),
                IndexRange(0, fern::size(result, 1)),
            }, value, result);
    }

};


template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value,
    class Result>
struct GradientXByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value,
    Result,
    ParallelExecutionPolicy,
    raster_2d_tag>
{

    // gradient(2d)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ParallelExecutionPolicy const& /* execution_policy */,
        Value const& value,
        Result& result)
    {
        assert(fern::size(value, 0) == fern::size(result, 0));
        assert(fern::size(value, 1) == fern::size(result, 1));

        ThreadPool& pool(ThreadClient::pool());
        size_t const size1 = fern::size(result, 0);
        size_t const size2 = fern::size(result, 1);
        std::vector<IndexRanges<2>> ranges = index_ranges(pool.size(),
            size1, size2);
        std::vector<std::future<void>> futures;
        futures.reserve(ranges.size());

        for(auto const& block_range: ranges) {
            auto function = std::bind(
                gradient_x_2d<InputNoDataPolicy, OutputNoDataPolicy,
                    Value, Result>,
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
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value,
    class Result,
    class ExecutionPolicy,
    class ValueCollectionCategory>
struct GradientYByArgumentCategory
{
};


template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value,
    class Result>
struct GradientYByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value,
    Result,
    SequentialExecutionPolicy,
    raster_2d_tag>
{

    // gradient(2d)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SequentialExecutionPolicy const& /* execution_policy */,
        Value const& value,
        Result& result)
    {
        assert(fern::size(value, 0) == fern::size(result, 0));
        assert(fern::size(value, 1) == fern::size(result, 1));

        gradient_y_2d(input_no_data_policy, output_no_data_policy,
            IndexRanges<2>{
                IndexRange(0, fern::size(result, 0)),
                IndexRange(0, fern::size(result, 1)),
            }, value, result);
    }

};


template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value,
    class Result>
struct GradientYByArgumentCategory<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value,
    Result,
    ParallelExecutionPolicy,
    raster_2d_tag>
{

    // gradient(2d)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ParallelExecutionPolicy const& /* execution_policy */,
        Value const& value,
        Result& result)
    {
        assert(fern::size(value, 0) == fern::size(result, 0));
        assert(fern::size(value, 1) == fern::size(result, 1));

        ThreadPool& pool(ThreadClient::pool());
        size_t const size1 = fern::size(result, 0);
        size_t const size2 = fern::size(result, 1);
        std::vector<IndexRanges<2>> ranges = index_ranges(pool.size(),
            size1, size2);
        std::vector<std::future<void>> futures;
        futures.reserve(ranges.size());

        for(auto const& block_range: ranges) {
            auto function = std::bind(
                gradient_y_2d<InputNoDataPolicy, OutputNoDataPolicy,
                    Value, Result>,
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
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value,
    class Result,
    class ExecutionPolicy>
class GradientX
{

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ExecutionPolicy const& execution_policy,
        Value const& value,
        Result& result)
    {

        GradientXByArgumentCategory<
            InputNoDataPolicy,
            OutputNoDataPolicy,
            Value,
            Result,
            ExecutionPolicy,
            base_class<argument_category<Value>, raster_2d_tag>>::apply(
                input_no_data_policy, output_no_data_policy, execution_policy,
                value, result);
    }

};


template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value,
    class Result>
struct GradientX<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value,
    Result,
    ExecutionPolicy>

{

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ExecutionPolicy const& execution_policy,
        Value const& value,
        Result& result)
    {
        switch(execution_policy.which()) {
            case fern::detail::sequential_execution_policy_id: {
                GradientXByArgumentCategory<
                    InputNoDataPolicy,
                    OutputNoDataPolicy,
                    Value,
                    Result,
                    SequentialExecutionPolicy,
                    base_class<argument_category<Value>, raster_2d_tag>>
                        ::apply(
                            input_no_data_policy, output_no_data_policy,
                            fern::detail::get_policy<SequentialExecutionPolicy>(
                                execution_policy), value, result);
                break;
            }
            case fern::detail::parallel_execution_policy_id: {
                GradientXByArgumentCategory<
                    InputNoDataPolicy,
                    OutputNoDataPolicy,
                    Value,
                    Result,
                    ParallelExecutionPolicy,
                    base_class<argument_category<Value>, raster_2d_tag>>
                        ::apply(
                            input_no_data_policy, output_no_data_policy,
                            fern::detail::get_policy<ParallelExecutionPolicy>(
                                execution_policy), value, result);
                break;
            }
        }
    }

};


template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value,
    class Result,
    class ExecutionPolicy>
class GradientY
{

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ExecutionPolicy const& execution_policy,
        Value const& value,
        Result& result)
    {

        GradientYByArgumentCategory<
            InputNoDataPolicy,
            OutputNoDataPolicy,
            Value,
            Result,
            ExecutionPolicy,
            base_class<argument_category<Value>, raster_2d_tag>>::apply(
                input_no_data_policy, output_no_data_policy, execution_policy,
                value, result);
    }

};


template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value,
    class Result>
struct GradientY<
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value,
    Result,
    ExecutionPolicy>

{

    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ExecutionPolicy const& execution_policy,
        Value const& value,
        Result& result)
    {
        switch(execution_policy.which()) {
            case fern::detail::sequential_execution_policy_id: {
                GradientYByArgumentCategory<
                    InputNoDataPolicy,
                    OutputNoDataPolicy,
                    Value,
                    Result,
                    SequentialExecutionPolicy,
                    base_class<argument_category<Value>, raster_2d_tag>>
                        ::apply(
                            input_no_data_policy, output_no_data_policy,
                            fern::detail::get_policy<SequentialExecutionPolicy>(
                                execution_policy), value, result);
                break;
            }
            case fern::detail::parallel_execution_policy_id: {
                GradientYByArgumentCategory<
                    InputNoDataPolicy,
                    OutputNoDataPolicy,
                    Value,
                    Result,
                    ParallelExecutionPolicy,
                    base_class<argument_category<Value>, raster_2d_tag>>
                        ::apply(
                            input_no_data_policy, output_no_data_policy,
                            fern::detail::get_policy<ParallelExecutionPolicy>(
                                execution_policy), value, result);
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
    class Result
>
void gradient_x(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    dispatch::GradientX<
        InputNoDataPolicy,
        OutputNoDataPolicy,
        Value,
        Result,
        ExecutionPolicy>::apply(
            input_no_data_policy, output_no_data_policy, execution_policy,
            value, result);
}


template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Value,
    class Result
>
void gradient_y(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    dispatch::GradientY<
        InputNoDataPolicy,
        OutputNoDataPolicy,
        Value,
        Result,
        ExecutionPolicy>::apply(
            input_no_data_policy, output_no_data_policy, execution_policy,
            value, result);
}

} // namespace detail
} // namespace gradient
} // namespace fern
