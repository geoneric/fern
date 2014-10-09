#pragma once
#include "fern/core/argument_traits.h"
#include "fern/algorithm/core/detail/cover.h"
#include "fern/algorithm/core/result_type.h"
#include "fern/algorithm/policy/policies.h"


namespace fern {
namespace algorithm {
namespace core {

/*!
    @ingroup    fern_algorithm_core_group
    @brief      Per position, select the first non-no-data element from
                @a value1 and @a value2, and write it to @a result.
    @sa         fern::algorithm::binary_local_operation

    - Value types of @a Value1, @a Value2 and @a Result must be arithmetic.
    - Value types of @a Value1, @a Value2 and @a Result must be the same.
    - The clone type of @a Result must equal the clone type of
      combining @a Value1 and @a Value2. See fern::algorithm::Result.
*/
template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename ExecutionPolicy,
    typename Value1,
    typename Value2,
    typename Result>
void cover(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy const& execution_policy,
    Value1 const& value1,
    Value2 const& value2,
    Result& result)
{

    // TODO
    // Cover supports passing combinations of arrays and constants. If arrays
    // are passed, the dimensionality must be the same. You can't pass a 2d and
    // a 1d array, but you can pass a 2d array and a constant.

    // TODO
    // The result must be a collection if one of the arguments is. Else
    // the result must be a constant.

    FERN_STATIC_ASSERT(std::is_same, value_type<Value1>, value_type<Value2>)
    FERN_STATIC_ASSERT(std::is_same, value_type<Result>, value_type<Value2>)
    FERN_STATIC_ASSERT(std::is_same, clone_type<Result, value_type<Result>>,
        clone_type<result_type<Value1, Value2>, value_type<Result>>)

    cover::detail::cover<>(input_no_data_policy,
        output_no_data_policy, execution_policy,
        value1, value2, result);
}


/*!
    @ingroup    fern_algorithm_core_group
    @overload
*/
template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename ExecutionPolicy,
    typename Value1,
    typename Value2,
    typename Result>
void cover(
    ExecutionPolicy const& execution_policy,
    Value1 const& value1,
    Value2 const& value2,
    Result& result)
{
    OutputNoDataPolicy output_no_data_policy;
    cover<>(InputNoDataPolicy(), output_no_data_policy,
        execution_policy, value1, value2, result);
}


/*!
    @ingroup    fern_algorithm_core_group
    @overload
*/
template<
    typename ExecutionPolicy,
    typename Value1,
    typename Value2,
    typename Result>
void cover(
    ExecutionPolicy const& execution_policy,
    Value1 const& value1,
    Value2 const& value2,
    Result& result)
{
    using InputNoDataPolicy = SkipNoData<SkipNoData<>, SkipNoData<>>;
    using OutputNoDataPolicy = DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;
    cover<>(InputNoDataPolicy(SkipNoData<>(), SkipNoData<>()),
        output_no_data_policy, execution_policy, value1, value2, result);
}

} // namespace core
} // namespace algorithm
} // namespace fern
