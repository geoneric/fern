// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/core/data_type_traits.h"
#include "fern/core/assert.h"
#include "fern/core/base_class.h"
#include "fern/core/data_customization_point.h"
#include "fern/algorithm/core/index_ranges.h"
#include "fern/algorithm/policy/execution_policy.h"


namespace fern {
namespace algorithm {
namespace detail {

template<
    typename Algorithm,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Result>
void operation_0d(
    Algorithm const& algorithm,
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    Result& result)
{
    if(std::get<0>(input_no_data_policy).is_no_data()) {
        output_no_data_policy.mark_as_no_data();
    }
    else {
        reference<Result> r(get(result));

        algorithm(r);

        /// if(!OutOfRangePolicy::within_range(r)) {
        ///     // Result value is out-of-range. Mark result value as
        ///     // no-data. Result value contains the out-of-range
        ///     // value (this may be overridden by
        ///     // output_no_data_policy, depending on its
        ///     // implementation).
        ///     output_no_data_policy.mark_as_no_data();
        /// }
    }
}


template<
    typename Algorithm,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Result>
void operation_1d(
    Algorithm const& algorithm,
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    IndexRanges<1> const& index_ranges,
    Result& result)
{
    for(size_t i = index_ranges[0].begin(); i < index_ranges[0].end(); ++i) {

        if(std::get<0>(input_no_data_policy).is_no_data(i)) {
            output_no_data_policy.mark_as_no_data(i);
        }
        else {

            reference<Result> r(get(result, i));

            algorithm(r);

            /// if(!OutOfRangePolicy::within_range(r)) {
            ///     // Result value is out-of-range. Mark result value as
            ///     // no-data. Result value contains the out-of-range
            ///     // value (this may be overridden by
            ///     // output_no_data_policy, depending on its
            ///     // implementation).
            ///     output_no_data_policy.mark_as_no_data(i);
            /// }
        }
    }
}


template<
    typename Algorithm,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Result>
void operation_2d(
    Algorithm const& algorithm,
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    IndexRanges<2> const& index_ranges,
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

                reference<Result> r(get(result, index_));

                algorithm(r);

                /// if(!OutOfRangePolicy::within_range(r)) {
                ///     // Result value is out-of-range. Mark result value as
                ///     // no-data. Result value contains the out-of-range
                ///     // value (this may be overridden by
                ///     // output_no_data_policy, depending on its
                ///     // implementation).
                ///     output_no_data_policy.mark_as_no_data(index_);
                /// }
            }

            ++index_;
        }
    }
}


namespace dispatch {

template<
    typename Algorithm,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Result,
    typename ExecutionPolicy,
    typename ValueCollectionCategory>
class NullaryLocalOperation
{
};


template<
    typename Algorithm,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Result,
    typename ExecutionPolicy>
struct NullaryLocalOperation<
    Algorithm,
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Result,
    ExecutionPolicy,
    array_0d_tag>

{

    // f(0d array)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ExecutionPolicy& /* execution_policy */,
        Result& result)
    {
        Algorithm algorithm;

        operation_0d<>(algorithm,
            input_no_data_policy, output_no_data_policy, result);
    }

};


template<
    typename Algorithm,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Result>
struct NullaryLocalOperation<
    Algorithm,
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Result,
    SequentialExecutionPolicy,
    array_1d_tag>

{

    // f(1d array)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SequentialExecutionPolicy& /* execution_policy */,
        Result& result)
    {
        Algorithm algorithm;

        operation_1d<>(algorithm,
            input_no_data_policy, output_no_data_policy,
            IndexRanges<1>{IndexRange(0, size(result))}, result);
    }

};


template<
    typename Algorithm,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Result>
struct NullaryLocalOperation<
    Algorithm,
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Result,
    ParallelExecutionPolicy,
    array_1d_tag>

{

    // f(1d array)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ParallelExecutionPolicy& execution_policy,
        Result& result)
    {
        ThreadPool& pool(execution_policy.thread_pool());
        size_t const size_ = size(result);
        std::vector<IndexRanges<1>> ranges = index_ranges(pool.size(), size_);
        std::vector<std::future<void>> futures;
        futures.reserve(ranges.size());

        Algorithm algorithm;

        for(auto const& block_range: ranges) {
            auto function = std::bind(
                operation_1d<
                    Algorithm,
                    InputNoDataPolicy, OutputNoDataPolicy,
                    Result>,
                std::cref(algorithm),
                std::cref(input_no_data_policy),
                std::ref(output_no_data_policy), std::cref(block_range),
                std::ref(result));
            futures.emplace_back(pool.submit(function));
        }

        for(auto& future: futures) {
            future.get();
        }
    }

};


template<
    typename Algorithm,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Result>
struct NullaryLocalOperation<
    Algorithm,
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Result,
    SequentialExecutionPolicy,
    array_2d_tag>

{

    // f(2d array)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        SequentialExecutionPolicy& /* execution_policy */,
        Result& result)
    {
        Algorithm algorithm;

        operation_2d<>(algorithm,
            input_no_data_policy, output_no_data_policy,
            IndexRanges<2>{
                IndexRange(0, size(result, 0)),
                IndexRange(0, size(result, 1))
            }, result);
    }

};


template<
    typename Algorithm,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Result>
struct NullaryLocalOperation<
    Algorithm,
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Result,
    ParallelExecutionPolicy,
    array_2d_tag>

{

    // f(2d array)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ParallelExecutionPolicy& execution_policy,
        Result& result)
    {
        ThreadPool& pool(execution_policy.thread_pool());
        size_t const size1 = size(result, 0);
        size_t const size2 = size(result, 1);
        std::vector<IndexRanges<2>> ranges = index_ranges(pool.size(),
            size1, size2);
        std::vector<std::future<void>> futures;
        futures.reserve(ranges.size());

        Algorithm algorithm;

        for(auto const& block_range: ranges) {
            auto function = std::bind(
                operation_2d<
                    Algorithm,
                    InputNoDataPolicy, OutputNoDataPolicy,
                    Result>,
                std::cref(algorithm),
                std::cref(input_no_data_policy),
                std::ref(output_no_data_policy), std::cref(block_range),
                std::ref(result));
            futures.emplace_back(pool.submit(function));
        }

        for(auto& future: futures) {
            future.get();
        }
    }

};


template<
    typename Algorithm,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename Result>
struct NullaryLocalOperation<
    Algorithm,
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Result,
    ExecutionPolicy,
    array_2d_tag>

{

    // f(2d array)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ExecutionPolicy& execution_policy,
        Result& result)
    {
        switch(execution_policy.which()) {
            case detail::sequential_execution_policy_id: {
                detail::dispatch::NullaryLocalOperation<
                    Algorithm,
                    InputNoDataPolicy,
                    OutputNoDataPolicy,
                    Result,
                    SequentialExecutionPolicy,
                    array_2d_tag>::apply(
                        input_no_data_policy, output_no_data_policy,
                        boost::get<SequentialExecutionPolicy>(execution_policy),
                        result);
                break;
            }
            case detail::parallel_execution_policy_id: {
                detail::dispatch::NullaryLocalOperation<
                    Algorithm,
                    InputNoDataPolicy,
                    OutputNoDataPolicy,
                    Result,
                    ParallelExecutionPolicy,
                    array_2d_tag>::apply(
                        input_no_data_policy, output_no_data_policy,
                        boost::get<ParallelExecutionPolicy>(execution_policy),
                        result);
                break;
            }
        }
    }

};

} // namespace dispatch
} // namespace detail


/*!
    @ingroup    fern_algorithm_core_group
    @brief      Function that executes a nullary local operation.
    @tparam     Algorithm Class of the operation to execute.
    @param[out] result Output that is written by the operation.
    @sa         fern::algorithm::unary_local_operation,
                fern::algorithm::binary_local_operation,
                fern::algorithm::n_ary_local_operation

    This function supports handling 0d, 1d and 2d values.

    This function supports sequential and parallel execution of the operation.
*/
template<
    typename Algorithm,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename ExecutionPolicy,
    typename Result
>
void nullary_local_operation(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy& execution_policy,
    Result& result)
{
    detail::dispatch::NullaryLocalOperation<
        Algorithm,
        InputNoDataPolicy,
        OutputNoDataPolicy,
        Result,
        ExecutionPolicy,
        base_class<argument_category<Result>, array_2d_tag>>::apply(
            input_no_data_policy, output_no_data_policy, execution_policy,
            result);
}

} // namespace algorithm
} // namespace fern
