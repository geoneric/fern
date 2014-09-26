#pragma once
/// #include "fern/core/assert.h"
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/core/detail/unite_no_data.h"


namespace fern {
namespace core {

/// //! Offset the elements in \a value by \a offset_ and write the result to \a result.
/// /*!
///     \ingroup       core
///     \sa            @ref fern_algorithm_core
/// 
///     Elements in \a result that have no corresponding value in \a value, will
///     be handled by \a output_no_data_policy.
/// 
///     A positive offset will offset the elements towards higher indices. In case
///     of a 1D array, for example, a positive offset will offset the elements
///     towards the end of the array.
/// 
///     - The dimensionality of \a value must be larger than 0.
///     - The \a offset_ must have the same dimensionality as \a value.
///     - The value type of \a Result must be equal to \a Value.
///     - The value type of \a Offset must be signed integral.
/// */
template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Value1,
    class Value2,
    class Result>
void unite_no_data(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy const& execution_policy,
    Value1 const& value1,
    Value2 const& value2,
    Result& result)
{
    /// FERN_STATIC_ASSERT(std::is_same, value_type<Result>, value_type<Value>)
    /// FERN_STATIC_ASSERT(std::is_integral, value_type<Offset>)
    /// FERN_STATIC_ASSERT(std::is_signed, value_type<Offset>)
    /// static_assert(rank<Value>() > 0, "Rank must be larger than 0");
    /// static_assert(rank<Value>() == rank<Offset>(),
    ///     "Rank of offset must equal rank of value");

    unite_no_data::detail::unite_no_data<>(input_no_data_policy,
        output_no_data_policy, execution_policy, value1, value2, result);
}


/*!
    \ingroup       core
    \overload
*/
template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Value1,
    class Value2,
    class Result>
void unite_no_data(
    ExecutionPolicy const& execution_policy,
    Value1 const& value1,
    Value2 const& value2,
    Result& result)
{
    OutputNoDataPolicy output_no_data_policy;
    unite_no_data<>(InputNoDataPolicy(), output_no_data_policy,
        execution_policy, value1, value2, result);
}


/*!
    \ingroup       core
    \overload
*/
template<
    class ExecutionPolicy,
    class Value1,
    class Value2,
    class Result>
void unite_no_data(
    ExecutionPolicy const& execution_policy,
    Value1 const& value1,
    Value2 const& value2,
    Result& result)
{
    using InputNoDataPolicy = SkipNoData<SkipNoData<>, SkipNoData<>>;
    using OutputNoDataPolicy = DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;
    unite_no_data<>(InputNoDataPolicy(SkipNoData<>(), SkipNoData<>()),
        output_no_data_policy, execution_policy, value1, value2, result);
}

} // namespace core
} // namespace fern
