#pragma once
#include "fern/algorithm/statistic/sum.h"
#include "fern/algorithm/statistic/detail/count.h"


namespace fern {
namespace statistic {

/// // Check for out of range results is not needed, assuming the result type is
/// // capable of storing at least the number of values in the input.
/// template<
///     class Values,
///     class Result,
///     class InputNoDataPolicy=SkipNoData,
///     class OutputNoDataPolicy=DontMarkNoData
/// >
/// class Count
/// {
/// 
/// public:
/// 
///     using category = local_aggregate_operation_tag;
///     using A = Values;
///     using AValue = typename ArgumentTraits<A>::value_type;
///     using R = Result;
///     using RValue = typename ArgumentTraits<R>::value_type;
/// 
///     FERN_STATIC_ASSERT(std::is_arithmetic, AValue)
///     FERN_STATIC_ASSERT(std::is_same, RValue, size_t)
/// 
///     Count()
///         : _algorithm()
///     {
///     }
/// 
///     Count(
///         InputNoDataPolicy&& input_no_data_policy,  // Universal reference.
///         OutputNoDataPolicy&& output_no_data_policy)  // Universal reference.
///         : _algorithm(
///               std::forward<InputNoDataPolicy>(input_no_data_policy),
///               std::forward<OutputNoDataPolicy>(output_no_data_policy))
///     {
///     }
/// 
///     inline void operator()(
///         A const& values,
///         AValue const& value,
///         R& result)
///     {
///         _algorithm.calculate(values, value, result);
///     }
/// 
///     template<
///         class Indices>
///     inline void operator()(
///         Indices const& indices,
///         A const& values,
///         AValue const& value,
///         R& result)
///     {
///         _algorithm.calculate(indices, values, value, result);
///     }
/// 
///     template<
///         class InputNoDataPolicy_,
///         class Collection>
///     inline void aggregate(
///         InputNoDataPolicy_&& input_no_data_policy,  // Universal reverence.
///         Collection const& results,
///         R& result)
///     {
///         Sum<Collection, R, binary::DiscardRangeErrors, InputNoDataPolicy_,
///             OutputNoDataPolicy>(
///                 std::forward<InputNoDataPolicy_>(input_no_data_policy),
///                 std::forward<OutputNoDataPolicy>(_algorithm))(results, result);
///     }
/// 
/// private:
/// 
///     count::detail::dispatch::Count<A, R,
///         InputNoDataPolicy, OutputNoDataPolicy,
///         typename ArgumentTraits<A>::argument_category> _algorithm;
/// 
/// };
/// 
/// 
/// //! Count the number of occurences of \a value in \a values and store the result in \a result.
/// /*!
///   \tparam    Values Type of \a values.
///   \tparam    Result Type of \a result.
///   \param     values Values to look in.
///   \param     value Value to look for.
///   \param     result Place to store the resulting count.
/// */
/// template<
///     class Values,
///     class Result,
///     class InputNoDataPolicy=SkipNoData,
///     class OutputNoDataPolicy=DontMarkNoData
/// >
/// void count(
///     Values const& values,
///     typename ArgumentTraits<Values>::value_type const& value,
///     Result& result)
/// {
///     Count<Values, Result, InputNoDataPolicy, OutputNoDataPolicy>()(values,
///         value, result);
/// }
/// 
/// 
/// //! Count the number of occurences of \a value in \a values and store the result in \a result.
/// /*!
///   \tparam    Values Type of \a values.
///   \tparam    Result Type of \a result.
///   \param     input_no_data_policy Input no-data policy.
///   \param     output_no_data_policy Output no-data policy.
///   \param     values Values to look in.
///   \param     value Value to look for.
///   \param     result Place to store the resulting count.
/// */
/// template<
///     class Values,
///     class Result,
///     class InputNoDataPolicy=SkipNoData,
///     class OutputNoDataPolicy=DontMarkNoData
/// >
/// void count(
///     InputNoDataPolicy&& input_no_data_policy,  // Universal reference.
///     OutputNoDataPolicy&& output_no_data_policy,  // Universal reference.
///     Values const& values,
///     typename ArgumentTraits<Values>::value_type const& value,
///     Result& result)
/// {
///     Count<Values, Result, InputNoDataPolicy, OutputNoDataPolicy>(
///         std::forward<InputNoDataPolicy>(input_no_data_policy),
///         std::forward<OutputNoDataPolicy>(output_no_data_policy))(
///             values, value, result);
/// }


template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Value,
    class Result
>
void count(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy const& execution_policy,
    Value const& values,
    value_type<Value> const& value,
    Result& result)
{
    FERN_STATIC_ASSERT(std::is_arithmetic, value_type<Value>)
    /// FERN_STATIC_ASSERT(!std::is_same, value_type<Value>, bool)
    FERN_STATIC_ASSERT(std::is_arithmetic, value_type<Result>)

    count::detail::count<>(input_no_data_policy, output_no_data_policy,
        execution_policy, values, value, result);
}


/*!
    \overload
*/
template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Value,
    class Result
>
void count(
    ExecutionPolicy const& execution_policy,
    Value const& values,
    value_type<Value> const& value,
    Result& result)
{
    OutputNoDataPolicy output_no_data_policy;
    count<>(InputNoDataPolicy(), output_no_data_policy, execution_policy,
        values, value, result);
}


/*!
    \overload
*/
template<
    class ExecutionPolicy,
    class Value,
    class Result
>
void count(
    ExecutionPolicy const& execution_policy,
    Value const& values,
    value_type<Value> const& value,
    Result& result)
{
    using InputNoDataPolicy = SkipNoData;
    using OutputNoDataPolicy = DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;
    count<>(InputNoDataPolicy(), output_no_data_policy, execution_policy,
        values, value, result);
}

} // namespace statistic
} // namespace fern
