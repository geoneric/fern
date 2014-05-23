#pragma once
#include <utility>
#include "fern/core/argument_categories.h"
#include "fern/core/argument_traits.h"
#include "fern/core/assert.h"
#include "fern/core/collection_traits.h"
#include "fern/algorithm/core/index_ranges.h"


namespace fern {
namespace unary_min {
namespace detail {
namespace dispatch {

template<class Values, class Result,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ArgumentCollectionCategory>
class UnaryMin
{
};


template<class Values, class Result,
    class InputNoDataPolicy,
    class OutputNoDataPolicy>
class UnaryMin<Values, Result,
        InputNoDataPolicy,
        OutputNoDataPolicy,
        array_0d_tag>:

    public InputNoDataPolicy,
    public OutputNoDataPolicy

{

    FERN_STATIC_ASSERT(std::is_same, Values, Result)

public:

    UnaryMin()
        : InputNoDataPolicy(),
          OutputNoDataPolicy()
    {
    }

    UnaryMin(
        InputNoDataPolicy&& input_no_data_policy,
        OutputNoDataPolicy&& output_no_data_policy)
        : InputNoDataPolicy(std::forward<InputNoDataPolicy>(
              input_no_data_policy)),
          OutputNoDataPolicy(std::forward<OutputNoDataPolicy>(
              output_no_data_policy))
    {
    }

    // constant
    inline void calculate(
        Values const& values,
        Result& result)
    {
        using INDP = InputNoDataPolicy;
        using ONDP = OutputNoDataPolicy;

        if(!INDP::is_no_data()) {
            fern::get(result) = fern::get(values);
        }
        else {
            ONDP::mark_as_no_data();
        }
    }

};


template<class Values, class Result,
    class InputNoDataPolicy,
    class OutputNoDataPolicy>
class UnaryMin<Values, Result,
        InputNoDataPolicy,
        OutputNoDataPolicy,
        array_1d_tag>:

    public InputNoDataPolicy,
    public OutputNoDataPolicy

{

    FERN_STATIC_ASSERT(std::is_same, value_type<Values>, Result)

public:

    UnaryMin()
        : InputNoDataPolicy(),
          OutputNoDataPolicy()
    {
    }

    UnaryMin(
        InputNoDataPolicy&& input_no_data_policy,
        OutputNoDataPolicy&& output_no_data_policy)
        : InputNoDataPolicy(std::forward<InputNoDataPolicy>(
              input_no_data_policy)),
          OutputNoDataPolicy(std::forward<OutputNoDataPolicy>(
              output_no_data_policy))
    {
    }

    // 1d array
    inline void calculate(
        Values const& values,
        Result& result)
    {
        size_t const size = fern::size(values);

        auto ranges = IndexRanges<1>{
            IndexRange(0, size)
        };

        calculate(ranges, values, result);
    }

    template<
        class Indices>
    inline void calculate(
        Indices const& indices,
        Values const& values,
        Result& result)
    {
        using INDP = InputNoDataPolicy;
        using ONDP = OutputNoDataPolicy;

        size_t const begin = indices[0].begin();
        size_t const end = indices[0].end();
        bool data_seen{false};

        if(begin < end) {

            for(size_t i = begin; i < end; ++i) {

                if(!INDP::is_no_data(i)) {

                    // Initialize result with first value.
                    result = fern::get(values, i);
                    data_seen = true;

                    for(++i; i < end; ++i) {

                        if(!INDP::is_no_data(i)) {
                            // Update result with minimum of current value
                            // and this new value.
                            result = std::min(result, fern::get(values, i));
                        }
                    }
                }
            }
        }

        if(!data_seen) {
            ONDP::mark_as_no_data();
        }
    }

};


template<class Values, class Result,
    class InputNoDataPolicy,
    class OutputNoDataPolicy>
class UnaryMin<Values, Result,
        InputNoDataPolicy,
        OutputNoDataPolicy,
        array_2d_tag>:

    public InputNoDataPolicy,
    public OutputNoDataPolicy

{

    FERN_STATIC_ASSERT(std::is_same, value_type<Values>, Result)

public:

    UnaryMin()
        : InputNoDataPolicy(),
          OutputNoDataPolicy()
    {
    }

    UnaryMin(
        InputNoDataPolicy&& input_no_data_policy,
        OutputNoDataPolicy&& output_no_data_policy)
        : InputNoDataPolicy(std::forward<InputNoDataPolicy>(
              input_no_data_policy)),
          OutputNoDataPolicy(std::forward<OutputNoDataPolicy>(
              output_no_data_policy))
    {
    }

    // 2d array
    inline void calculate(
        Values const& values,
        Result& result)
    {
        size_t const size1 = fern::size(values, 0);
        size_t const size2 = fern::size(values, 1);

        auto ranges = IndexRanges<2>{
            IndexRange(0, size1),
            IndexRange(0, size2)
        };

        calculate(ranges, values, result);
    }

    template<
        class Indices>
    inline void calculate(
        Indices const& indices,
        Values const& values,
        Result& result)
    {
        using INDP = InputNoDataPolicy;
        using ONDP = OutputNoDataPolicy;

        size_t const begin1 = indices[0].begin();
        size_t const end1 = indices[0].end();
        size_t const begin2 = indices[1].begin();
        size_t const end2 = indices[1].end();
        bool data_seen{false};

        if(begin1 < end1 && begin2 < end2) {

            for(size_t i = begin1; i < end1; ++i) {
                for(size_t j = begin2; j < end2; ++j) {

                    if(!INDP::is_no_data(i, j)) {

                        // Initialize result with first value.
                        result = fern::get(values, i, j);
                        data_seen = true;

                        for(; i < end1; ++i) {
                            for(++j; j < end2; ++j) {

                                if(!INDP::is_no_data(i, j)) {
                                    // Update result with minimum of current
                                    // value and this new value.
                                    result = std::min(result,
                                        fern::get(values, i, j));
                                }
                            }
                        }
                    }
                }
            }
        }

        if(!data_seen) {
            ONDP::mark_as_no_data();
        }
    }

};

} // namespace dispatch
} // namespace detail
} // namespace unary_min
} // namespace fern
