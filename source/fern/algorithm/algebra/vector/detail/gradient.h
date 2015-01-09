#pragma once
#include <algorithm>
#include "fern/core/base_class.h"
#include "fern/core/raster_traits.h"
#include "fern/core/thread_client.h"
#include "fern/algorithm/policy/execution_policy.h"
#include "fern/algorithm/core/index_ranges.h"


namespace fern {
namespace algorithm {
namespace gradient {
namespace detail {

template<
    typename Value,
    typename Distance>
inline constexpr Value gradient(
    Value const& value1,
    Value const& value2,
    Distance const& distance)
{
    return (value2 - value1) / distance;
}


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result>
static void gradient_x_2d(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    IndexRanges<2> const& index_ranges,
    Value const& value,
    Result& result)
{
    size_t const size2{size(value, 1)};
    double const distance1{cell_size(value, 0)};
    double const distance2{distance1 + distance1};

    size_t index_left, index_center, index_right;

    // Handle left border, in case it is within the index range.
    for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end(); ++i) {

        index_center = index(result, i, index_ranges[1].begin());
        index_right = index_center + 1;

        for(size_t j = index_ranges[1].begin(); j < 1; ++j) {

            // Left cell of window lies outside of the raster. Consider it
            // no-data.

            if(std::get<0>(input_no_data_policy).is_no_data(index_center)) {
                output_no_data_policy.mark_as_no_data(index_center);
            }
            else {
                assert(j + 1 < size2);

                if(!std::get<0>(input_no_data_policy).is_no_data(index_right)) {
                    // x c r
                    get(result, index_center) = gradient(
                        get(value, index_center),
                        get(value, index_right),
                        distance1);
                }
                else  {
                    // x c x
                    get(result, index_center) = 0;
                }
            }

            ++index_center;
            ++index_right;
        }
    }


    // Handle right border, in case it is within the index range.
    for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end(); ++i) {

        index_center = index(result, i, size2 - 1);
        assert(index_center > 0);
        index_left = index_center - 1;

        for(size_t j = size2 - 1; j < index_ranges[1].end(); ++j) {

            // Right cell of window lies outside of the raster. Consider it
            // no-data.

            if(std::get<0>(input_no_data_policy).is_no_data(index_center)) {
                output_no_data_policy.mark_as_no_data(index_center);
            }
            else {
                assert(j > 0);

                if(!std::get<0>(input_no_data_policy).is_no_data(index_left)) {
                    // l c x
                    get(result, index_center) = gradient(
                        get(value, index_left),
                        get(value, index_center),
                        distance1);
                }
                else  {
                    // x c x
                    get(result, index_center) = 0;
                }
            }

            ++index_center;
            ++index_left;
        }
    }


    // Handle innert part, except for the borders (in case borders are within
    // the index range).
    for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end(); ++i) {

        index_center = index(result, i, std::max<size_t>(1u,
            index_ranges[1].begin()));
        index_left = index_center - 1;
        index_right = index_center + 1;

        for(size_t j = std::max<size_t>(1u, index_ranges[1].begin());
                j < std::min<size_t>(size2 - 1u, index_ranges[1].end()); ++j) {

            // All cells of window lie within the raster.

            if(std::get<0>(input_no_data_policy).is_no_data(index_center)) {
                output_no_data_policy.mark_as_no_data(index_center);
            }
            else {
                assert(j > 0);
                assert(j + 1 < size2);

                if(!std::get<0>(input_no_data_policy).is_no_data(index_left)) {
                    if(!std::get<0>(input_no_data_policy).is_no_data(
                            index_right)) {
                        // l c r
                        get(result, index_center) = gradient(
                            get(value, index_left),
                            get(value, index_right),
                            distance2);
                    }
                    else {
                        // l c x
                        get(result, index_center) = gradient(
                            get(value, index_left),
                            get(value, index_center),
                            distance1);
                    }
                }
                else if(!std::get<0>(input_no_data_policy).is_no_data(
                        index_right)) {
                    // x c r
                    get(result, index_center) = gradient(
                        get(value, index_center),
                        get(value, index_right),
                        distance1);
                }
                else {
                    // x c x
                    get(result, index_center) = 0;
                }
            }

            ++index_center;
            ++index_left;
            ++index_right;
        }
    }
}


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result>
static void gradient_y_2d(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    IndexRanges<2> const& index_ranges,
    Value const& value,
    Result& result)
{
    size_t const size1{size(value, 0)};
    double const distance1{cell_size(value, 1)};
    double const distance2{distance1 + distance1};

    size_t index_top, index_center, index_bottom;

    // Handle top border, in case it is within the index range.
    for(size_t i = index_ranges[0].begin(); i < 1; ++i) {

        index_center = index(result, i, index_ranges[1].begin());
        index_bottom = index(result, i + 1, index_ranges[1].begin());

        for(size_t j = index_ranges[1].begin(); j < index_ranges[1].end();
            ++j) {

            // Top cell of window lies outside of the raster. Consider it
            // no-data.

            if(std::get<0>(input_no_data_policy).is_no_data(index_center)) {
                output_no_data_policy.mark_as_no_data(index_center);
            }
            else {
                assert(i + 1 < size1);

                if(!std::get<0>(input_no_data_policy).is_no_data(
                        index_bottom)) {
                    // x c r
                    get(result, index_center) = gradient(
                        get(value, index_center),
                        get(value, index_bottom),
                        distance1);
                }
                else  {
                    // x c x
                    get(result, index_center) = 0;
                }
            }

            ++index_center;
            ++index_bottom;
        }
    }


    // Handle bottom border, in case it is within the index range.
    for(size_t i = size1 - 1; i < index_ranges[0].end(); ++i) {

        index_center = index(result, i, index_ranges[1].begin());
        index_top = index(result, i - 1, index_ranges[1].begin());

        for(size_t j = index_ranges[1].begin(); j < index_ranges[1].end();
                ++j) {

            // Bottom cell of window lies outside of the raster. Consider it
            // no-data.

            if(std::get<0>(input_no_data_policy).is_no_data(index_center)) {
                output_no_data_policy.mark_as_no_data(index_center);
            }
            else {
                assert(i > 0);

                if(!std::get<0>(input_no_data_policy).is_no_data(index_top)) {
                    // l c x
                    get(result, index_center) = gradient(
                        get(value, index_top),
                        get(value, index_center),
                        distance1);
                }
                else  {
                    // x c x
                    get(result, index_center) = 0;
                }
            }

            ++index_center;
            ++index_top;
        }
    }


    // Handle innert part, except for the borders (in case borders are within
    // the index range).
    for(size_t i = std::max<size_t>(1u, index_ranges[0].begin());
            i < std::min<size_t>(size1 - 1u, index_ranges[0].end()); ++i) {

        index_center = index(result, i, index_ranges[1].begin());
        index_bottom = index(result, i + 1, index_ranges[1].begin());
        index_top = index(result, i - 1, index_ranges[1].begin());

        for(size_t j = index_ranges[1].begin(); j < index_ranges[1].end();
                ++j) {

            // All cells of window lie within the raster.

            if(std::get<0>(input_no_data_policy).is_no_data(index_center)) {
                output_no_data_policy.mark_as_no_data(index_center);
            }
            else {
                assert(i > 0);
                assert(i + 1 < size1);

                if(!std::get<0>(input_no_data_policy).is_no_data(index_top)) {
                    if(!std::get<0>(input_no_data_policy).is_no_data(
                            index_bottom)) {
                        // l c r
                        get(result, index_center) = gradient(
                            get(value, index_top),
                            get(value, index_bottom),
                            distance2);
                    }
                    else {
                        // l c x
                        get(result, index_center) = gradient(
                            get(value, index_top),
                            get(value, index_center),
                            distance1);
                    }
                }
                else if(!std::get<0>(input_no_data_policy).is_no_data(
                        index_bottom)) {
                    // x c r
                    get(result, index_center) = gradient(
                        get(value, index_center),
                        get(value, index_bottom),
                        distance1);
                }
                else {
                    // x c x
                    get(result, index_center) = 0;
                }
            }

            ++index_center;
            ++index_top;
            ++index_bottom;
        }
    }
}


namespace dispatch {

template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result,
    typename ExecutionPolicy,
    typename ValueCollectionCategory>
struct GradientXByArgumentCategory
{
};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result>
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
        assert(size(value, 0) == size(result, 0));
        assert(size(value, 1) == size(result, 1));

        gradient_x_2d(input_no_data_policy, output_no_data_policy,
            IndexRanges<2>{
                IndexRange(0, size(result, 0)),
                IndexRange(0, size(result, 1)),
            }, value, result);
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result>
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
        assert(size(value, 0) == size(result, 0));
        assert(size(value, 1) == size(result, 1));

        ThreadPool& pool(ThreadClient::pool());
        size_t const size1 = size(result, 0);
        size_t const size2 = size(result, 1);
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
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result,
    typename ExecutionPolicy,
    typename ValueCollectionCategory>
struct GradientYByArgumentCategory
{
};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result>
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
        assert(size(value, 0) == size(result, 0));
        assert(size(value, 1) == size(result, 1));

        gradient_y_2d(input_no_data_policy, output_no_data_policy,
            IndexRanges<2>{
                IndexRange(0, size(result, 0)),
                IndexRange(0, size(result, 1)),
            }, value, result);
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result>
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
        assert(size(value, 0) == size(result, 0));
        assert(size(value, 1) == size(result, 1));

        ThreadPool& pool(ThreadClient::pool());
        size_t const size1 = size(result, 0);
        size_t const size2 = size(result, 1);
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
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result,
    typename ExecutionPolicy>
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
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result>
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
            case fern::algorithm::detail::sequential_execution_policy_id: {
                GradientXByArgumentCategory<
                    InputNoDataPolicy,
                    OutputNoDataPolicy,
                    Value,
                    Result,
                    SequentialExecutionPolicy,
                    base_class<argument_category<Value>, raster_2d_tag>>
                        ::apply(
                            input_no_data_policy, output_no_data_policy,
                            fern::algorithm::detail::get_policy<
                                SequentialExecutionPolicy>(execution_policy),
                            value, result);
                break;
            }
            case fern::algorithm::detail::parallel_execution_policy_id: {
                GradientXByArgumentCategory<
                    InputNoDataPolicy,
                    OutputNoDataPolicy,
                    Value,
                    Result,
                    ParallelExecutionPolicy,
                    base_class<argument_category<Value>, raster_2d_tag>>
                        ::apply(
                            input_no_data_policy, output_no_data_policy,
                            fern::algorithm::detail::get_policy<
                                ParallelExecutionPolicy>(execution_policy),
                            value, result);
                break;
            }
        }
    }

};


template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result,
    typename ExecutionPolicy>
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
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Value,
    typename Result>
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
            case fern::algorithm::detail::sequential_execution_policy_id: {
                GradientYByArgumentCategory<
                    InputNoDataPolicy,
                    OutputNoDataPolicy,
                    Value,
                    Result,
                    SequentialExecutionPolicy,
                    base_class<argument_category<Value>, raster_2d_tag>>
                        ::apply(
                            input_no_data_policy, output_no_data_policy,
                            fern::algorithm::detail::get_policy<
                                SequentialExecutionPolicy>(execution_policy),
                            value, result);
                break;
            }
            case fern::algorithm::detail::parallel_execution_policy_id: {
                GradientYByArgumentCategory<
                    InputNoDataPolicy,
                    OutputNoDataPolicy,
                    Value,
                    Result,
                    ParallelExecutionPolicy,
                    base_class<argument_category<Value>, raster_2d_tag>>
                        ::apply(
                            input_no_data_policy, output_no_data_policy,
                            fern::algorithm::detail::get_policy<
                                ParallelExecutionPolicy>(execution_policy),
                            value, result);
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
    typename Result
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
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename ExecutionPolicy,
    typename Value,
    typename Result
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
} // namespace algorithm
} // namespace fern
