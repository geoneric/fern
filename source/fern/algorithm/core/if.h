#pragma once
#include "fern/core/assert.h"
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/core/detail/if.h"
#include "fern/algorithm/core/result_type.h"


namespace fern {
namespace algorithm {
namespace core {

/*!
    @ingroup    fern_algorithm_core_group
    @brief      Conditionally assign @a true_value to @a result.

    All elements in @a condition that evaluate to true are assigned to the
    @a result. All other elements in result are handled by the
    @a output_no_data policy.

    - The value type of @a Condition must be arithmetic.
    - The value type of @a Result must be equal to @a TrueValue.
    - The clone type of @a Result must equal the clone type of
      combining @a Condition and @a TrueValue. See fern::algorithm::Result.
*/
template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename ExecutionPolicy,
    typename Condition,
    typename TrueValue,
    typename Result>
void if_(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy& execution_policy,
    Condition const& condition,
    TrueValue const& true_value,
    Result& result)
{
    FERN_STATIC_ASSERT(std::is_arithmetic, value_type<Condition>)
    FERN_STATIC_ASSERT(std::is_same, value_type<Result>, value_type<TrueValue>)
    FERN_STATIC_ASSERT(std::is_same, CloneT<Result, value_type<Result>>,
        CloneT<result_type<Condition, TrueValue>, value_type<Result>>)

    if_::detail::if_<>(input_no_data_policy,
        output_no_data_policy, execution_policy, condition, true_value, result);
}


/*!
    @ingroup    fern_algorithm_core_group
    @overload
*/
template<
    typename ExecutionPolicy,
    typename Condition,
    typename TrueValue,
    typename Result>
void if_(
    ExecutionPolicy& execution_policy,
    Condition const& condition,
    TrueValue const& true_value,
    Result& result)
{
    using InputNoDataPolicy = InputNoDataPolicies<SkipNoData, SkipNoData>;
    using OutputNoDataPolicy = DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;
    if_<>(InputNoDataPolicy{{}, {}}, output_no_data_policy, execution_policy,
        condition, true_value, result);
}


/*!
    @ingroup    fern_algorithm_core_group
    @brief      Conditionally assign @a true_value or @a false_value to
                @a result.

    All elements in @a condition that evaluate to true are assigned the
    corresponding element from @a true_value. All elements in @a condition
    that evaluate to false are assigned the corresponding element from
    @a false_value.

    - The value type of @a Condition must be arithmetic.
    - The value type of @a TrueValue must equal the value type of
      @a FalseValue.
    - The value type of @a Result must be equal to @a TrueValue.
    - The clone type of @a Result must equal the clone type of
      combining @a Condition, @a TrueValue and @a FalseValue. See
      fern::algorithm::Result.
*/
template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename ExecutionPolicy,
    typename Condition,
    typename TrueValue,
    typename FalseValue,
    typename Result>
void if_(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy& execution_policy,
    Condition const& condition,
    TrueValue const& true_value,
    FalseValue const& false_value,
    Result& result)
{
    FERN_STATIC_ASSERT(std::is_arithmetic, value_type<Condition>)
    FERN_STATIC_ASSERT(std::is_same, value_type<TrueValue>,
        value_type<FalseValue>)
    FERN_STATIC_ASSERT(std::is_same, value_type<Result>, value_type<TrueValue>)
    FERN_STATIC_ASSERT(std::is_same, CloneT<Result, value_type<Result>>,
        CloneT<result_type<result_type<Condition, TrueValue>, FalseValue>,
        value_type<Result>>)

    if_::detail::if_<>(input_no_data_policy,
        output_no_data_policy, execution_policy, condition, true_value,
        false_value, result);
}


/*!
    @ingroup    fern_algorithm_core_group
    @overload
*/
template<
    typename ExecutionPolicy,
    typename Condition,
    typename TrueValue,
    typename FalseValue,
    typename Result>
void if_(
    ExecutionPolicy& execution_policy,
    Condition const& condition,
    TrueValue const& true_value,
    FalseValue const& false_value,
    Result& result)
{
    using InputNoDataPolicy = InputNoDataPolicies<SkipNoData, SkipNoData,
          SkipNoData>;
    using OutputNoDataPolicy = DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;
    if_<>(InputNoDataPolicy{{}, {}, {}}, output_no_data_policy,
        execution_policy, condition, true_value, false_value, result);
}

} // namespace core
} // namespace algorithm
} // namespace fern
