#pragma once
#include "fern/core/assert.h"
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/algebra/vector/detail/lax.h"


namespace fern {

namespace algebra {

//! Calculate the lax of \a value and write the result to \a result.
/*!
    The algorithm implemented is similar to [the one implemented in PCRaster]
    (https://sourceforge.net/p/pcraster/pcrtree2/ci/master/tree/sources/calc/vf.c).

    In short/pseudo code, the algorithm:

    \code
    result = (1 - fraction) * value + fraction * convolution(value, kernel);
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

    The value type of \a Value and \a Result must be floating point and the
    same.

    \ingroup       vector
    \sa            @ref fern_algorithm_algebra_vector
*/
template<
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Value,
    class Result>
void lax(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Value const& fraction,
    Result& result)
{
    FERN_STATIC_ASSERT(std::is_floating_point, value_type<Value>)
    FERN_STATIC_ASSERT(std::is_same, value_type<Result>, value_type<Value>)

    lax::detail::lax(input_no_data_policy,
        output_no_data_policy, execution_policy, value, fraction, result);
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
void lax(
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Value const& fraction,
    Result& result)
{
    OutputNoDataPolicy output_no_data_policy;
    lax(InputNoDataPolicy(), output_no_data_policy, execution_policy,
        value, fraction, result);
}


/*!
    \ingroup       vector
    \overload
*/
template<
    class ExecutionPolicy,
    class Value,
    class Result>
void lax(
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Value const& fraction,
    Result& result)
{
    using InputNoDataPolicy = SkipNoData;
    using OutputNoDataPolicy = DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;
    lax(InputNoDataPolicy(), output_no_data_policy, execution_policy,
        value, fraction, result);
}

} // namespace algebra
} // namespace fern


/// /* lax(): a window average filter similar to Lax scheme, used to avoid
///  * numerical instability
///  * */
/// int vf_lax(
///     MAP_REAL8* result,          // 2d
///     MAP_REAL8 const* input,     // 2d
///     MAP_REAL8 const* fractMap)  // 0d
/// {
///     double cellv, othercellv, gg, hh;
/// 
///     // get the nonspatial
///     double frac;
///     fractMap->Get(&frac, 0, 0, fractMap);
/// 
///     int nrows  = result->NrRows(result);
///     int ncols  = result->NrCols(result);
/// 
///     // pseudo code:
///     // convolution(raster, kernel, result);
///     // times(result, fraction, result);
///     // times(raster, 1 - fraction, times_result);
///     // add(result, times_result, result);
/// 
///     // -> Iterate over all cells.
///     for(int r = 0; r < nrows; ++r) {
///         for(int c = 0; c < ncols; ++c) {
/// 
///             gg = 0;
///             hh = 0;
/// 
///             // -> Current cell value in cellv.
///             if(input->Get(&cellv, r, c, input) &&
///                    fractMap->Get(&frac, r, c, fractMap)) {
/// 
///                 // -> Get upper left value.
///                 if(input->Get(&othercellv, r - 1, c - 1, input)) {
///                     gg = gg + 2 * othercellv;
///                     hh = hh + 2;
///                 }
/// 
///                 if(input->Get(&othercellv, r - 1, c + 0, input)) {
///                     gg = gg + 3 * othercellv;
///                     hh = hh + 3;
///                 }
/// 
///                 if(input->Get(&othercellv, r - 1, c + 1, input)) {
///                     gg = gg + 2 * othercellv;
///                     hh = hh + 2;
///                 }
/// 
/// 
///                 if(input->Get(&othercellv, r + 0, c - 1, input)) {
///                     gg = gg + 3 * othercellv;
///                     hh = hh + 3;
///                 }
/// 
///                 if(input->Get(&othercellv, r + 0, c + 1, input)) {
///                     gg = gg + 3 * othercellv;
///                     hh = hh + 3;
///                 }
/// 
/// 
///                 if(input->Get(&othercellv, r + 1, c - 1, input)) {
///                     gg = gg + 2 * othercellv;
///                     hh = hh + 2;
///                 }
/// 
///                 if(input->Get(&othercellv, r + 1, c + 0, input)) {
///                     gg = gg + 3 * othercellv;
///                     hh = hh + 3;
///                 }
/// 
///                 if(input->Get(&othercellv, r + 1, c + 1, input)) {
///                     gg = gg + 2 * othercellv;
///                     hh = hh + 2;
///                 }
/// 
///                 result->Put(
///                     (1 - frac) * cellv +
///                     frac * (gg / hh),
///                     r, c, result);
///             }
///             else {
///                 result->PutMV(r, c, result);
///             }
///         }
///     }
/// 
///     return 0;
/// }
