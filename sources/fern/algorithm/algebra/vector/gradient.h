#pragma once
#include "fern/core/assert.h"
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/algebra/vector/detail/gradient.h"


namespace fern {
namespace algebra {

//! Calculate the gradient in x of \a value and write the result to \a result.
/*!
    The algorithm implemented is similar to [the one implemented in PCRaster]
    (https://sourceforge.net/p/pcraster/pcrtree2/ci/master/tree/sources/calc/vf.c).

    - The value type of \a value must be floating point.
    - Value type of \a result must equal the value type of \a result.
    - \a value and \a result must be rasters.

    \ingroup       vector
    \sa            @ref fern_algorithm_algebra_vector
*/
template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Value,
    class Result>
void gradient_x(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy const& execution_policy,
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
    \ingroup       vector
    \overload
*/
template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Value,
    class Result>
void gradient_x(
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    OutputNoDataPolicy output_no_data_policy;
    gradient_x(InputNoDataPolicy(), output_no_data_policy, execution_policy,
        value, result);
}


/*!
    \ingroup       vector
    \overload
*/
template<
    class ExecutionPolicy,
    class Value,
    class Result>
void gradient_x(
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    using InputNoDataPolicy = SkipNoData;
    using OutputNoDataPolicy = DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;
    gradient_x(InputNoDataPolicy(), output_no_data_policy, execution_policy,
        value, result);
}

} // namespace algebra
} // namespace fern


/// /* gradx(): calculates the directional derivative (gradient)
///  * of an input_raster field f(x,y) in the x-wise direction:
///  * gradx(f) = d/dx(f)
///  * */
/// 
/// 
/// extern int vf_gradx(
///     MAP_REAL8* result_raster,
///     MAP_REAL8 const* input_raster)
/// {
///     double value, right, left;
///     int nrows = result_raster->NrRows(result_raster);
///     int ncols = result_raster->NrCols(result_raster);
///     double dx = input_raster->CellLength(input_raster);
/// 
///     for(int r = 0; r < nrows; ++r) {
/// 
///         for(int c = 0; c < ncols; ++c) {
/// 
///             if(input_raster->Get(&value, r, c, input_raster)) {
///                 // Center value exists.
/// 
///                 if(input_raster->Get(&right, r, c + 1, input_raster) &&
///                         input_raster->Get(&left,  r, c - 1, input_raster) ) {
///                     // l c r
///                     // Left, center and right value exist.
///                     // Use second order central difference to get the
///                     // derivative.
///                     result_raster->Put((right - left) / (2 * dx),
///                         r, c, result_raster);
///                 }
///                 else if(input_raster->Get(&right, r, c + 1, input_raster)) {
///                     // x c r
///                     // Center and right value exist.
///                     // Use first order finite difference to get the
///                     // derivative.
///                     result_raster->Put((right - value) / dx,
///                         r, c, result_raster);
///                 }
///                 else if(input_raster->Get(&left, r, c - 1, input_raster))  {
///                     // l c x
///                     // Left and center value exist.
///                     // Use first order finite difference to get the
///                     // derivative.
///                     result_raster->Put((value - left) / dx,
///                         r, c, result_raster);
///                 }
///                 else {
///                     // Cell is isolated, the derivative is zero.
///                     result_raster->Put(0,
///                         r, c, result_raster);
///                 }
///             }
///             else {
///                 // ? x ?
///                 result_raster->PutMV(r, c, result_raster);
///             }
///         }
///     }
/// 
///     return 0;
/// }
