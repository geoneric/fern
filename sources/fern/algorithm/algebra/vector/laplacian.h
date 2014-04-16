#pragma once
#include "fern/core/assert.h"
#include "fern/core/argument_traits.h"
#include "fern/algorithm/core/operation_categories.h"
#include "fern/algorithm/policy/discard_domain_errors.h"
#include "fern/algorithm/policy/discard_range_errors.h"
#include "fern/algorithm/policy/dont_mark_no_data.h"


// https://en.wikipedia.org/wiki/Vector_calculus

namespace fern {
namespace laplacian {

//! The out-of-domain policy for the laplacian operation.
/*!
  \tparam    A Type of argument.

  All values are within the domain of valid values for laplacian.
*/
template<
    class A>
using OutOfDomainPolicy = DiscardDomainErrors;


//! The out-of-range policy for the laplacian operation.
/*!
  The result of the laplacian operation is a floating point. This policy
  verifies whether the result value is finite.
*/
class OutOfRangePolicy
{

public:

    template<
        class R>
    static constexpr bool within_range(
        R const& result)
    {
        FERN_STATIC_ASSERT(std::is_floating_point, R)

        return std::isfinite(result);
    }

};


/// template<
///     class A1,
///     class A2>
/// struct Algorithm
/// {
///     FERN_STATIC_ASSERT(std::is_arithmetic, A1)
///     FERN_STATIC_ASSERT(std::is_arithmetic, A2)
/// 
///     template<
///         class R>
///     inline void operator()(
///         A1 const& argument1,
///         A2 const& argument2,
///         R& result) const
///     {
///         result = static_cast<R>(argument1) + static_cast<R>(argument2);
///     }
/// 
/// };

} // namespace laplacian


namespace algebra {

template<
    class A,
    class OutOfDomainPolicy=DiscardDomainErrors,
    class OutOfRangePolicy=DiscardRangeErrors,
    class NoDataPolicy=DontMarkNoData
>
struct Laplacian
{

    using category = neighborhood_operation_tag;

    //! Type of the result of the operation.
    using R = A;

    using AValue = typename ArgumentTraits<A>::value_type;

    Laplacian()
        // : algorithm(plus::Algorithm<A1Value, A2Value>())
    {
    }

    explicit Laplacian(
        NoDataPolicy&& no_data_policy)
        // : algorithm(std::forward<NoDataPolicy>(no_data_policy),
        //     plus::Algorithm<A1Value, A2Value>())
    {
    }

    inline void operator()(
        A const& /* argument */,
        R& /* result */)
    {
        // algorithm.calculate(argument1, argument2, result);
    }

    // Laplacian as implemented in PCRaster is
    // - A convolution with this kernel:
    //   +---+---+---+
    //   | 2 | 3 | 2 |
    //   +---+---+---+
    //   | 3 | 0 | 3 |
    //   +---+---+---+
    //   | 2 | 3 | 2 |
    //   +---+---+---+
    //   - But without dividing by the sum of weights!
    //   - In case no-data is encountered, the value of the center is used.
    //     Not sure if we want that. Maybe optional. FilterPolicy.
    //     I would think that no data and their corresponding filter weights
    //     must be skipped.
    // - Substracted by the sum_of_weights * center value.
    // - Divided by dx * dx.
    //
    // convolute(argument, kernel, result);
    // result = (result - (sum(kernel) * argument)) / (dx * dx);
    //
    // So, we require:
    // - [*] sum: Add all values in a kernel. None of these are no-data.
    // - [ ] substract: Substract two 2D arrays.
    // - [ ] multiply: Multiply a constant and a 2D array.
    // - [ ] divide: Divice a 2D array by a constant.
    // - [*] convolute: Convolute a 2D array by a kernel.
    //
    // Alternative is to do everything in one operation: laplacian. This will
    // be a bit more efficient, since it is more cache friendly. Let's first
    // create one based on basic operations and optimize if needed (compare
    // with PCRaster).
    //
    // We need dx, which is the cell length. So we need something else to be
    // passed in than a 2D array.

    // template<
    //     class Indices>
    // inline void operator()(
    //     Indices const& indices,
    //     A1 const& argument1,
    //     A2 const& argument2,
    //     R& result)
    // {
    //     algorithm.calculate(indices, argument1, argument2, result);
    // }

    // detail::dispatch::BinaryOperation<A1, A2, R,
    //     OutOfDomainPolicy, OutOfRangePolicy, NoDataPolicy,
    //     plus::Algorithm<
    //         typename ArgumentTraits<A1>::value_type,
    //         typename ArgumentTraits<A2>::value_type>,
    //     typename ArgumentTraits<A1>::argument_category,
    //     typename ArgumentTraits<A2>::argument_category> algorithm;

};


//! Calculate the result of adding \a argument1 to \a argument2 and put it in \a result.
/*!
  \tparam    A1 Type of \a argument1.
  \tparam    A2 Type of \a argument2.
  \param     argument1 First argument to add.
  \param     argument2 Second argument to add.
  \return    Result is stored in argument \a result.

  This function uses the Plus class template with default policies for handling
  out-of-domain values, out-of-range values and no-data.
*/
template<
    class A>
void laplacian(
    A const& argument,
    typename Laplacian<A>::R& result)
{
    Laplacian<A>()(argument, result);
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
