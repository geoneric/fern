#pragma once
#include "fern/algorithm/statistic/sum.h"


namespace fern {
namespace count {
namespace detail {
namespace dispatch {

template<class Values, class Result,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ArrayCollectionCategory>
class Count
{
};


template<class Values, class Result,
    class InputNoDataPolicy,
    class OutputNoDataPolicy>
class Count<Values, Result,
        InputNoDataPolicy,
        OutputNoDataPolicy,
        array_0d_tag>:

    public InputNoDataPolicy,
    public OutputNoDataPolicy

{

public:

    Count()
        : InputNoDataPolicy(),
          OutputNoDataPolicy()
    {
    }

    Count(
        InputNoDataPolicy&& input_no_data_policy,
        OutputNoDataPolicy&& output_no_data_policy)
        : InputNoDataPolicy(std::forward<InputNoDataPolicy>(
              input_no_data_policy)),
          OutputNoDataPolicy(std::forward<OutputNoDataPolicy>(
              output_no_data_policy))
    {
    }

    // 0d array
    inline void calculate(
        Values const& values,
        typename ArgumentTraits<Values>::value_type const& value,
        Result& result)
    {
        if(!InputNoDataPolicy::is_no_data()) {
            fern::get(result) = fern::get(values) == fern::get(value) ? 1 : 0;
        }
        else {
            OutputNoDataPolicy::mark_as_no_data();
        }
    }

};


template<class Values, class Result,
    class InputNoDataPolicy,
    class OutputNoDataPolicy>
class Count<Values, Result,
        InputNoDataPolicy,
        OutputNoDataPolicy,
        array_1d_tag>:

    public InputNoDataPolicy,
    public OutputNoDataPolicy

{

public:

    Count()
        : InputNoDataPolicy(),
          OutputNoDataPolicy()
    {
    }

    Count(
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
        typename ArgumentTraits<Values>::value_type const& value,
        Result& result)
    {
        size_t const size = fern::size(values);

        auto ranges = IndexRanges<1>{
            IndexRange(0, size)
        };

        calculate(ranges, values, value, result);
    }

    template<
        class Indices>
    inline void calculate(
        Indices const& indices,
        Values const& values,
        typename ArgumentTraits<Values>::value_type const& value,
        Result& result)
    {
        size_t const begin = indices[0].begin();
        size_t const end = indices[0].end();
        bool data_seen{false};

        if(begin < end) {

            typename ArgumentTraits<Result>::value_type& count =
                fern::get(result);
            count = 0;

            for(size_t i = begin; i < end; ++i) {

                if(!InputNoDataPolicy::is_no_data(i)) {

                    count += fern::get(values, i) == value ? 1 : 0;
                    data_seen = true;
                }
            }
        }

        if(!data_seen) {
            OutputNoDataPolicy::mark_as_no_data();
        }
    }

};


template<class Values, class Result,
    class InputNoDataPolicy,
    class OutputNoDataPolicy>
class Count<Values, Result,
        InputNoDataPolicy,
        OutputNoDataPolicy,
        array_2d_tag>:

    public InputNoDataPolicy,
    public OutputNoDataPolicy

{

public:

    Count()
        : InputNoDataPolicy(),
          OutputNoDataPolicy()
    {
    }

    Count(
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
        typename ArgumentTraits<Values>::value_type const& value,
        Result& result)
    {
        size_t const size1 = fern::size(values, 0);
        size_t const size2 = fern::size(values, 1);

        auto ranges = IndexRanges<2>{
            IndexRange(0, size1),
            IndexRange(0, size2)
        };

        calculate(ranges, values, value, result);
    }

    template<
        class Indices>
    inline void calculate(
        Indices const& indices,
        Values const& values,
        typename ArgumentTraits<Values>::value_type const& value,
        Result& result)
    {
        size_t const begin1 = indices[0].begin();
        size_t const end1 = indices[0].end();
        size_t const begin2 = indices[1].begin();
        size_t const end2 = indices[1].end();
        bool data_seen{false};

        if(begin1 < end1 && begin2 < end2) {

            typename ArgumentTraits<Result>::value_type& count =
                fern::get(result);
            count = 0;

            for(size_t i = begin1; i < end1; ++i) {
                for(size_t j = begin2; j < end2; ++j) {

                    if(!InputNoDataPolicy::is_no_data(i, j)) {

                        count += fern::get(values, i, j) == value ? 1 : 0;
                        data_seen = true;
                    }
                }
            }
        }

        if(!data_seen) {
            OutputNoDataPolicy::mark_as_no_data();
        }
    }

};

} // namespace dispatch
} // namespace detail
} // namespace count


namespace statistic {

// Check for out of range results is not needed, assuming the result type is
// capable of storing at least the number of values in the input.
template<
    class Values,
    class Result,
    class InputNoDataPolicy=SkipNoData,
    class OutputNoDataPolicy=DontMarkNoData
>
class Count
{

public:

    using category = local_aggregate_operation_tag;
    using A = Values;
    using AValue = typename ArgumentTraits<A>::value_type;
    using R = Result;
    using RValue = typename ArgumentTraits<R>::value_type;

    FERN_STATIC_ASSERT(std::is_arithmetic, AValue)
    FERN_STATIC_ASSERT(std::is_same, RValue, size_t)

    Count()
        : _algorithm()
    {
    }

    Count(
        InputNoDataPolicy&& input_no_data_policy,  // Universal reference.
        OutputNoDataPolicy&& output_no_data_policy)  // Universal reference.
        : _algorithm(
              std::forward<InputNoDataPolicy>(input_no_data_policy),
              std::forward<OutputNoDataPolicy>(output_no_data_policy))
    {
    }

    inline void operator()(
        A const& values,
        AValue const& value,
        R& result)
    {
        _algorithm.calculate(values, value, result);
    }

    template<
        class Indices>
    inline void operator()(
        Indices const& indices,
        A const& values,
        AValue const& value,
        R& result)
    {
        _algorithm.calculate(indices, values, value, result);
    }

    template<
        class InputNoDataPolicy_,
        class Collection>
    inline void aggregate(
        InputNoDataPolicy_&& input_no_data_policy,  // Universal reverence.
        Collection const& results,
        R& result)
    {
        Sum<Collection, R, binary::DiscardRangeErrors, InputNoDataPolicy_,
            OutputNoDataPolicy>(
                std::forward<InputNoDataPolicy_>(input_no_data_policy),
                std::forward<OutputNoDataPolicy>(_algorithm))(results, result);
    }

private:

    count::detail::dispatch::Count<A, R,
        InputNoDataPolicy, OutputNoDataPolicy,
        typename ArgumentTraits<A>::argument_category> _algorithm;

};


//! Count the number of occurences of \a value in \a values and store the result in \a result.
/*!
  \tparam    Values Type of \a values.
  \tparam    Result Type of \a result.
  \param     values Values to look in.
  \param     value Value to look for.
  \param     result Place to store the resulting count.
*/
template<
    class Values,
    class Result,
    class InputNoDataPolicy=SkipNoData,
    class OutputNoDataPolicy=DontMarkNoData
>
void count(
    Values const& values,
    typename ArgumentTraits<Values>::value_type const& value,
    Result& result)
{
    Count<Values, Result, InputNoDataPolicy, OutputNoDataPolicy>()(values,
        value, result);
}


//! Count the number of occurences of \a value in \a values and store the result in \a result.
/*!
  \tparam    Values Type of \a values.
  \tparam    Result Type of \a result.
  \param     input_no_data_policy Input no-data policy.
  \param     output_no_data_policy Output no-data policy.
  \param     values Values to look in.
  \param     value Value to look for.
  \param     result Place to store the resulting count.
*/
template<
    class Values,
    class Result,
    class InputNoDataPolicy=SkipNoData,
    class OutputNoDataPolicy=DontMarkNoData
>
void count(
    InputNoDataPolicy&& input_no_data_policy,  // Universal reference.
    OutputNoDataPolicy&& output_no_data_policy,  // Universal reference.
    Values const& values,
    typename ArgumentTraits<Values>::value_type const& value,
    Result& result)
{
    Count<Values, Result, InputNoDataPolicy, OutputNoDataPolicy>(
        std::forward<InputNoDataPolicy>(input_no_data_policy),
        std::forward<OutputNoDataPolicy>(output_no_data_policy))(
            values, value, result);
}

} // namespace statistic
} // namespace fern
