#pragma once
#include "fern/core/assert.h"
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/algebra/vector/detail/laplacian.h"


namespace fern {
namespace laplacian {

//! The out-of-range policy for the laplacian operation.
/*!
  The result of the laplacian operation is a floating point. This policy
  verifies whether the result value is finite.
*/
template<
    class Value,
    class Result>
class OutOfRangePolicy
{

    FERN_STATIC_ASSERT(std::is_arithmetic, Result)
    FERN_STATIC_ASSERT(std::is_floating_point, Result)

public:

    inline bool within_range(
        Value const& /* value */,
        Result const& result) const
    {
        return std::isfinite(result);
    }

};

} // namespace laplacian


namespace algebra {

//! Calculate the laplacian of \a value and write the result to \a result.
/*!
    The algorithm implemented is similar to [the one implemented in PCRaster]
    (https://sourceforge.net/p/pcraster/pcrtree2/ci/master/tree/sources/calc/vf.c).

    In short/pseudo code, the algorithm:

    \code
    convolve(value, kernel, result);
    result = (result - (sum(kernel) * value)) / cell_area;
    result = ((convolve(value, kernel) -
        (convolve(defined(value), kernel(1)) * value)) / cell_area;
    \endcode

    Kernel:

    \code
    +---+---+---+
    | 2 | 3 | 2 |
    +---+---+---+
    | 3 | 0 | 3 |
    +---+---+---+
    | 2 | 3 | 2 |
    +---+---+---+
    \endcode

    The value type of \a value and \a result must be floating point and the
    same.

    \ingroup       vector
*/
template<
    template<class, class> class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Value,
    class Result>
void laplacian(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    FERN_STATIC_ASSERT(std::is_arithmetic, value_type<Value>)
    FERN_STATIC_ASSERT(std::is_floating_point, value_type<Result>)
    FERN_STATIC_ASSERT(std::is_same, value_type<Value>, value_type<Result>)

    laplacian::detail::laplacian<OutOfRangePolicy>(input_no_data_policy,
        output_no_data_policy, execution_policy, value, result);
}


/*!
    \ingroup       vector
    \overload
*/
template<
    template<class, class> class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Value,
    class Result>
void laplacian(
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    OutputNoDataPolicy output_no_data_policy;
    laplacian<OutOfRangePolicy>(InputNoDataPolicy(), output_no_data_policy,
        execution_policy, value, result);
}


/*!
    \ingroup       vector
    \overload
*/
template<
    class ExecutionPolicy,
    class Value,
    class Result>
void laplacian(
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    using InputNoDataPolicy = SkipNoData;
    using OutputNoDataPolicy = DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;
    laplacian<unary::DiscardRangeErrors>(InputNoDataPolicy(),
        output_no_data_policy, execution_policy, value, result);
}

} // namespace algebra
} // namespace fern



/// /* calculates the laplacian (div dot grad)
///  * of a scalar field f(x,y): laplacian(f) = div · grad(f) = d^2(f)/dx^2
///  * */
/// extern int vf_laplacian(MAP_REAL8 *result,
///                   const MAP_REAL8 *scalar)
/// {
///     int nrows, ncols;
///     double dx, value, neighbour, gg;
///     nrows  = result->NrRows(result);
///     ncols  = result->NrCols(result);
///     dx     = scalar->CellLength(scalar);
/// 
///     for(int r = 0; r < nrows; ++r) {
///         for(int c = 0; c < ncols; ++c) {
///             gg = 0;
/// 
///             // gg becomes sum of:
///             //     2 * (north-west or center)
///             //     2 * (north-east or center)
///             //     2 * (south-west or center)
///             //     2 * (south-east or center)
///             //     3 * (north or center)
///             //     3 * (west or center)
///             //     3 * (east or center)
///             //     3 * (south or center)
///             //
///             // These are 20 values.
///             //
///             // result becomes
///             // (gg - (20 * center)) / (dx * dx)
/// 
///             if(scalar->Get(&value, r + 0, c + 0, scalar)) {
///                 // North-west cell.
///                 if(scalar->Get(&neighbour, r - 1, c - 1, scalar)) {
///                     gg = gg + 2 * neighbour;
///                 }
///                 else {
///                     gg = gg + 2 * value;
///                 }
/// 
///                 // North cell.
///                 if(scalar->Get(&neighbour, r - 1, c + 0, scalar)) {
///                     gg = gg + 3 * neighbour;
///                 }
///                 else  gg = gg + 3 * value;
/// 
///                 // North-east cell.
///                 if(scalar->Get(&neighbour, r - 1, c + 1, scalar)) {
///                     gg = gg + 2 * neighbour;
///                 }
///                 else {
///                     gg = gg + 2 * value;
///                 }
/// 
///                 // West cell.
///                 if(scalar->Get(&neighbour, r + 0, c - 1, scalar)) {
///                     gg = gg + 3 * neighbour;
///                 }
///                 else {
///                     gg = gg + 3 * value;
///                 }
/// 
///                 // East cell.
///                 if(scalar->Get(&neighbour, r + 0, c + 1, scalar)) {
///                     gg = gg + 3 * neighbour;
///                 }
///                 else {
///                     gg = gg + 3 * value;
///                 }
/// 
///                 // South-west cell.
///                 if(scalar->Get(&neighbour, r + 1, c - 1, scalar)) {
///                     gg = gg + 2*neighbour;
///                 }
///                 else {
///                     gg = gg + 2 * value;
///                 }
/// 
///                 // South cell.
///                 if(scalar->Get(&neighbour, r + 1, c + 0, scalar)) {
///                     gg = gg + 3 * neighbour;
///                 }
///                 else {
///                     gg = gg + 3 * value;
///                 }
/// 
///                 // South-east cell.
///                 if(scalar->Get(&neighbour, r + 1, c + 1, scalar)) {
///                     gg = gg + 2 * neighbour;
///                 }
///                 else {
///                     gg = gg + 2 * value;
///                 }
/// 
///                 result->Put((gg - 20 * value) / (dx * dx), r, c, result);
///             }
///             else {
///                 result->PutMV(r, c, result);
///             }
///         }
///    }
/// 
///    return 0;
/// }
