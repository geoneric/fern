#pragma once
#include "fern/core/assert.h"
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/algebra/vector/detail/gradient.h"


namespace fern {
namespace algorithm {
namespace algebra {

/*!
    @ingroup    fern_algorithm_algebra_vector_group
    @brief      Calculate the gradient in x of @a value and write the result
                to @a result.

    The algorithm implemented is similar to [the one implemented in PCRaster]
    (https://sourceforge.net/p/pcraster/pcrtree2/ci/master/tree/sources/calc/vf.c).

    - The value type of @a value must be floating point.
    - Value type of @a result must equal the value type of @a result.
    - @a value and @a result must be rasters.
*/
template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename ExecutionPolicy,
    typename Value,
    typename Result>
void gradient_x(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy& execution_policy,
    Value const& value,
    Result& result)
{
    FERN_STATIC_ASSERT(std::is_floating_point, value_type<Value>)
    FERN_STATIC_ASSERT(std::is_same, value_type<Result>, value_type<Value>)
    FERN_STATIC_ASSERT(std::is_same, argument_category<Value>, raster_2d_tag)
    FERN_STATIC_ASSERT(std::is_same, argument_category<Value>, raster_2d_tag)

    gradient::detail::gradient_x(input_no_data_policy, output_no_data_policy,
        execution_policy, value, result);
}


/*!
    @ingroup    fern_algorithm_algebra_vector_group
    @overload
*/
template<
    typename ExecutionPolicy,
    typename Value,
    typename Result>
void gradient_x(
    ExecutionPolicy& execution_policy,
    Value const& value,
    Result& result)
{
    using InputNoDataPolicy = InputNoDataPolicies<SkipNoData>;
    using OutputNoDataPolicy = DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;
    gradient_x(InputNoDataPolicy{{}}, output_no_data_policy, execution_policy,
        value, result);
}


/*!
    @ingroup    fern_algorithm_algebra_vector_group
    @brief      Calculate the gradient in y of @a value and write the result
                to @a result.

    The algorithm implemented is similar to [the one implemented in PCRaster]
    (https://sourceforge.net/p/pcraster/pcrtree2/ci/master/tree/sources/calc/vf.c).

    - The value type of @a value must be floating point.
    - Value type of @a result must equal the value type of @a result.
    - @a value and @a result must be rasters.
*/
template<
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename ExecutionPolicy,
    typename Value,
    typename Result>
void gradient_y(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy& execution_policy,
    Value const& value,
    Result& result)
{
    FERN_STATIC_ASSERT(std::is_floating_point, value_type<Value>)
    FERN_STATIC_ASSERT(std::is_same, value_type<Result>, value_type<Value>)
    FERN_STATIC_ASSERT(std::is_same, argument_category<Value>, raster_2d_tag)
    FERN_STATIC_ASSERT(std::is_same, argument_category<Value>, raster_2d_tag)

    gradient::detail::gradient_y(input_no_data_policy, output_no_data_policy,
        execution_policy, value, result);
}


/*!
    @ingroup    fern_algorithm_algebra_vector_group
    @overload
*/
template<
    typename ExecutionPolicy,
    typename Value,
    typename Result>
void gradient_y(
    ExecutionPolicy& execution_policy,
    Value const& value,
    Result& result)
{
    using InputNoDataPolicy = InputNoDataPolicies<SkipNoData>;
    using OutputNoDataPolicy = DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;
    gradient_y(InputNoDataPolicy{{}}, output_no_data_policy, execution_policy,
        value, result);
}

} // namespace algebra
} // namespace algorithm
} // namespace fern
