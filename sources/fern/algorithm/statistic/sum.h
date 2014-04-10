#pragma once
#include "fern/core/assert.h"
#include "fern/algorithm/policy/discard_range_errors.h"
#include "fern/algorithm/policy/dont_mark_no_data.h"
#include "fern/algorithm/policy/skip_no_data.h"


namespace fern {
namespace sum {
namespace detail {
namespace dispatch {

template<class Argument, class Result,
    class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class A1CollectionCategory>
struct Sum
{
};


template<class Argument, class Result,
    class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy>
class Sum<Argument, Result,
        OutOfRangePolicy,
        InputNoDataPolicy,
        OutputNoDataPolicy,
        array_0d_tag>:

    public OutOfRangePolicy,
    public InputNoDataPolicy,
    public OutputNoDataPolicy

{

public:

    // This means that there cannot be overflow/underflow, so we don't need
    // to check for it.
    FERN_STATIC_ASSERT(std::is_same, Argument, Result)

    Sum()
        : OutOfRangePolicy(),
          InputNoDataPolicy(),
          OutputNoDataPolicy()
    {
    }

    Sum(
        InputNoDataPolicy&& input_no_data_policy,
        OutputNoDataPolicy&& output_no_data_policy)
        : OutOfRangePolicy(),
          InputNoDataPolicy(std::forward<InputNoDataPolicy>(
              input_no_data_policy)),
          OutputNoDataPolicy(std::forward<OutputNoDataPolicy>(
              output_no_data_policy))
    {
    }

    // constant
    inline void calculate(
        Argument const& argument,
        Result& result)
    {
        if(!InputNoDataPolicy::is_no_data()) {
            fern::get(result) = fern::get(argument);
        }
        else {
            OutputNoDataPolicy::mark_as_no_data();
        }
    }

private:

};


template<class Argument, class Result,
    class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy>
class Sum<Argument, Result,
        OutOfRangePolicy,
        InputNoDataPolicy,
        OutputNoDataPolicy,
        array_1d_tag>:

    public OutOfRangePolicy,
    public InputNoDataPolicy,
    public OutputNoDataPolicy

{

public:

    Sum()
        : OutOfRangePolicy(),
          InputNoDataPolicy(),
          OutputNoDataPolicy()
    {
    }

    Sum(
        InputNoDataPolicy&& input_no_data_policy,
        OutputNoDataPolicy&& output_no_data_policy)
        : OutOfRangePolicy(),
          InputNoDataPolicy(std::forward<InputNoDataPolicy>(
              input_no_data_policy)),
          OutputNoDataPolicy(std::forward<OutputNoDataPolicy>(
              output_no_data_policy))
    {
    }

    // 1d array
    inline void calculate(
        Argument const& argument,
        Result& result)
    {
        size_t const size = fern::size(argument);

        auto ranges = IndexRanges<1>{
            IndexRange(0, size)
        };

        calculate(ranges, argument, result);
    }

    template<
        class Indices>
    inline void calculate(
        Indices const& indices,
        Argument const& argument,
        Result& result)
    {
        size_t const begin = indices[0].begin();
        size_t const end = indices[0].end();
        bool data_seen{false};

        if(begin < end) {
            typename ArgumentTraits<Argument>::value_type value;
            typename ArgumentTraits<Result>::value_type tmp_result{0};
            fern::get(result) = tmp_result;

            for(size_t i = begin; i < end; ++i) {

                if(!InputNoDataPolicy::is_no_data(i)) {

                    value = fern::get(argument, i);
                    tmp_result += value;

                    // lhs, rhs, lhs + rhs
                    if(!OutOfRangePolicy::within_range(fern::get(result), value,
                            tmp_result)) {
                        OutputNoDataPolicy::mark_as_no_data();
                        break;
                    }

                    fern::get(result) = tmp_result;

                    data_seen = true;
                }
            }
        }

        if(!data_seen) {
            OutputNoDataPolicy::mark_as_no_data();
        }
    }

private:

};


template<class Argument, class Result,
    class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy>
class Sum<Argument, Result,
        OutOfRangePolicy,
        InputNoDataPolicy,
        OutputNoDataPolicy,
        array_2d_tag>:

    public OutOfRangePolicy,
    public InputNoDataPolicy,
    public OutputNoDataPolicy

{

public:

    Sum()
        : OutOfRangePolicy(),
          InputNoDataPolicy(),
          OutputNoDataPolicy()
    {
    }

    Sum(
        InputNoDataPolicy&& input_no_data_policy,
        OutputNoDataPolicy&& output_no_data_policy)
        : OutOfRangePolicy(),
          InputNoDataPolicy(std::forward<InputNoDataPolicy>(
              input_no_data_policy)),
          OutputNoDataPolicy(std::forward<OutputNoDataPolicy>(
              output_no_data_policy))
    {
    }

    // 2d array
    inline void calculate(
        Argument const& argument,
        Result& result)
    {
        size_t const size1 = fern::size(argument, 0);
        size_t const size2 = fern::size(argument, 1);

        auto ranges = IndexRanges<2>{
            IndexRange(0, size1),
            IndexRange(0, size2)
        };

        calculate(ranges, argument, result);
    }

    template<
        class Indices>
    inline void calculate(
        Indices const& indices,
        Argument const& argument,
        Result& result)
    {

        /// Instead of three nested loops, a single loop is sufficient to run
        /// from index 0 to (width*height*depth).
        ///
        /// long index = 0;
        /// const long indexend = (long)width * (long)height * (long)depth;
        /// while(index != indexend)
        ///     sum += data[index++];

        size_t const begin1 = indices[0].begin();
        size_t const end1 = indices[0].end();
        size_t const begin2 = indices[1].begin();
        size_t const end2 = indices[1].end();
        bool data_seen{false};

        if(begin1 < end1 && begin2 < end2) {

            typename ArgumentTraits<Argument>::value_type value;
            typename ArgumentTraits<Result>::value_type tmp_result{0};
            fern::get(result) = tmp_result;

            for(size_t i = begin1; i < end1; ++i) {
                for(size_t j = begin2; j < end2; ++j) {

                    if(!InputNoDataPolicy::is_no_data(i, j)) {

                        value = fern::get(argument, i, j);
                        tmp_result += value;

                        // lhs, rhs, lhs + rhs
                        if(!OutOfRangePolicy::within_range(fern::get(result),
                                value, tmp_result)) {
                            OutputNoDataPolicy::mark_as_no_data();
                            break;
                        }

                        fern::get(result) = tmp_result;

                        data_seen = true;
                    }
                }

                // TODO We need to break again! The inner breaks breaks only
                //      the inner for loop.
            }
        }

        if(!data_seen) {
            OutputNoDataPolicy::mark_as_no_data();
        }
    }

private:

};

} // namespace dispatch
} // namespace detail
} // namespace sum


namespace statistic {

//! Implementation of the sum operation.
/*!
  This operation calculates the result of adding all values in a collection.
*/
template<
    class Argument,
    class Result,
    class OutOfRangePolicy=DiscardRangeErrors,
    class InputNoDataPolicy=SkipNoData,
    class OutputNoDataPolicy=DontMarkNoData
>
class Sum
{

public:

    using category = local_aggregate_operation_tag;
    using A = Argument;
    using AValue = typename ArgumentTraits<A>::value_type;
    using R = Result;
    using RValue = typename ArgumentTraits<R>::value_type;

    FERN_STATIC_ASSERT(std::is_arithmetic, AValue)
    // We see boolean more as nominal values than as integers. You can't sum'em.
    FERN_STATIC_ASSERT(!std::is_same, AValue, bool)
    FERN_STATIC_ASSERT(std::is_same, RValue, AValue)

    //! Default constructor.
    /*!
    */
    Sum()
        : _algorithm()
    {
    }

    //! Constructor taking a no-data policy.
    /*!
      This constructor can be used to pass in a no-data policy that is not
      default-constructed.
    */
    Sum(
        InputNoDataPolicy&& input_no_data_policy,  // Universal reference.
        OutputNoDataPolicy&& output_no_data_policy)  // Universal reference.
        : _algorithm(
              std::forward<InputNoDataPolicy>(input_no_data_policy),
              std::forward<OutputNoDataPolicy>(output_no_data_policy))
    {
    }

    //! Perform the algorithm on \a argument and store the result in \a result.
    /*!
    */
    inline void operator()(
        Argument const& argument,
        Result& result)
    {
        _algorithm.calculate(argument, result);
    }


    template<
        class Indices>
    inline void operator()(
        Indices const& indices,
        Argument const& argument,
        Result& result)
    {
        _algorithm.calculate(indices, argument, result);
    }


    //! Aggregate the \a results per block into a final \a result.
    /*!
      \tparam    Collection Type of collection with results per block.
      \param     results Results per block.
      \param     result Final result.
    */
    template<
        class InputNoDataPolicy_,
        class Collection>
    inline void aggregate(
        InputNoDataPolicy_&& input_no_data_policy,  // Universal reverence.
        Collection const& results,
        Result& result)
    {
        Sum<Collection, Result, OutOfRangePolicy, InputNoDataPolicy_,
            OutputNoDataPolicy>(
                std::forward<InputNoDataPolicy_>(input_no_data_policy),
                std::forward<OutputNoDataPolicy>(_algorithm))(results, result);
    }

private:

    //! Algorithm used to calculate results.
    sum::detail::dispatch::Sum<Argument, Result,
        OutOfRangePolicy, InputNoDataPolicy, OutputNoDataPolicy,
        typename ArgumentTraits<Argument>::argument_category> _algorithm;

};


//! Sum the values in \a argument and store the result in \a result.
/*!
*/
template<
    class Argument,
    class Result,
    class OutOfRangePolicy=DiscardRangeErrors,
    class InputNoDataPolicy=SkipNoData,
    class OutputNoDataPolicy=DontMarkNoData
>
void sum(
    Argument const& argument,
    Result& result)
{
    Sum<Argument, Result, OutOfRangePolicy, InputNoDataPolicy,
        OutputNoDataPolicy>()(argument, result);
}


//! Sum the values in \a argument and store the result in \a result.
/*!
  Use this overload if you need to pass already constructed policies or
  policies which have non-default constuctors.
*/
template<
    class Argument,
    class Result,
    class OutOfRangePolicy=DiscardRangeErrors,
    class InputNoDataPolicy=SkipNoData,
    class OutputNoDataPolicy=DontMarkNoData
>
void sum(
    InputNoDataPolicy&& input_no_data_policy,  // Universal reference.
    OutputNoDataPolicy&& output_no_data_policy,  // Universal reference.
    Argument const& argument,
    Result& result)
{
    Sum<Argument, Result, OutOfRangePolicy, InputNoDataPolicy,
        OutputNoDataPolicy>(
            std::forward<InputNoDataPolicy>(input_no_data_policy),
            std::forward<OutputNoDataPolicy>(output_no_data_policy))(
                argument, result);
}

} // namespace statistic
} // namespace fern
