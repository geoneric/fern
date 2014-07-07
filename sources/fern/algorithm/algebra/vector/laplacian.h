#pragma once
/// #include "fern/core/argument_traits.h"
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

/// template<
///     class Values,
///     class Result,
///     template<class> class OutOfDomainPolicy=unary::DiscardDomainErrors,
///     template<class, class> class OutOfRangePolicy=unary::DiscardRangeErrors,
///     class InputNoDataPolicy=SkipNoData,
///     class OutputNoDataPolicy=DontMarkNoData
/// >
/// class Laplacian
/// {
/// 
/// public:
/// 
///     using category = focal_operation_tag;
///     using A = Values;
///     using AValue = value_type<A>;
///     using R = Result;
///     using RValue = value_type<R>;
/// 
///     FERN_STATIC_ASSERT(std::is_arithmetic, AValue)
///     FERN_STATIC_ASSERT(std::is_arithmetic, RValue)
///     FERN_STATIC_ASSERT(std::is_same, AValue, RValue)
/// 
///     Laplacian()
///         : _algorithm()
///     {
///     }
/// 
///     Laplacian(
///         InputNoDataPolicy&& input_no_data_policy,  // Universal reference.
///         OutputNoDataPolicy&& output_no_data_policy)  // Universal reference.
///         : _algorithm(
///             std::forward<InputNoDataPolicy>(input_no_data_policy),
///             std::forward<OutputNoDataPolicy>(output_no_data_policy))
///     {
///     }
/// 
///     inline void operator()(
///         A const& values,
///         R& result)
///     {
///         _algorithm.calculate(values, result);
///     }
/// 
///     template<
///         class Indices>
///     inline void operator()(
///         Indices const& indices,
///         A const& values,
///         R& result)
///     {
///         _algorithm.calculate(indices, values, result);
///     }
/// 
/// private:
/// 
///     // Laplacian as implemented in PCRaster is
///     // - A convolution with this kernel:
///     //   +---+---+---+
///     //   | 2 | 3 | 2 |
///     //   +---+---+---+
///     //   | 3 | 0 | 3 |
///     //   +---+---+---+
///     //   | 2 | 3 | 2 |
///     //   +---+---+---+
///     //   - But without dividing by the sum of weights! FilterPolicy.
///     //   - In case no-data is encountered, the value of the center is used.
///     //     Not sure if we want that. Maybe optional. FilterPolicy.
///     //     I would think that no data and their corresponding filter weights
///     //     must be skipped.
///     // - Substracted by the sum_of_weights * center value.
///     // - Divided by dx * dx.
///     //
///     // convolute(argument, kernel, result);
///     // result = (result - (sum(kernel) * argument)) / (dx * dx);
///     // result = ((convolve(arg, kernel) -
///     //              (convolve(defined(arg), kernel(1)) * arg)) / cell_area
///     //
///     // So, we require:
///     // - [*] sum: Add all values in a kernel. None of these are no-data.
///     // - [*] subtract: Subtract two 2D arrays.
///     // - [*] multiply: Multiply a constant and a 2D array.
///     // - [*] divide: Divide a 2D array by a constant.
///     // - [*] convolute: Convolute a 2D array by a kernel.
///     //
///     // Alternative is to do everything in one operation: laplacian. This will
///     // be a bit more efficient, since it is more cache friendly. Let's first
///     // create one based on basic operations and optimize if needed (compare
///     // with PCRaster).
///     //
///     // We need dx, which is the cell length. So we need something else to be
///     // passed in than a 2D array.
/// 
///     laplacian::detail::dispatch::Laplacian<A, R,
///         InputNoDataPolicy, OutputNoDataPolicy,
///         typename ArgumentTraits<A>::argument_category> _algorithm;
/// 
/// };
/// 
/// 
/// template<
///     class Values,
///     class Result,
///     template<class> class OutOfDomainPolicy=unary::DiscardDomainErrors,
///     template<class, class> class OutOfRangePolicy=unary::DiscardRangeErrors,
///     class InputNoDataPolicy=SkipNoData,
///     class OutputNoDataPolicy=DontMarkNoData
/// >
/// void laplacian(
///     Values const& values,
///     Result& result)
/// {
///     Laplacian<Values, Result, OutOfDomainPolicy, OutOfRangePolicy,
///         InputNoDataPolicy, OutputNoDataPolicy>()(values, result);
/// }
/// 
/// 
/// /*!
///   \overload
/// */
/// template<
///     class Values,
///     class Result,
///     template<class> class OutOfDomainPolicy=unary::DiscardDomainErrors,
///     template<class, class> class OutOfRangePolicy=unary::DiscardRangeErrors,
///     class InputNoDataPolicy=SkipNoData,
///     class OutputNoDataPolicy=DontMarkNoData
/// >
/// void laplacian(
///     InputNoDataPolicy&& input_no_data_policy,  // Universal reference.
///     OutputNoDataPolicy&& output_no_data_policy,  // Universal reference.
///     Values const& values,
///     Result& result)
/// {
///     Laplacian<Values, Result, OutOfDomainPolicy, OutOfRangePolicy,
///         InputNoDataPolicy, OutputNoDataPolicy>(
///             std::forward<InputNoDataPolicy>(input_no_data_policy),
///             std::forward<OutputNoDataPolicy>(output_no_data_policy))(
///         values, result);
/// }


template<
    template<class, class> class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Value,
    class Result
>
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


template<
    template<class, class> class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Value,
    class Result
>
void laplacian(
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    OutputNoDataPolicy output_no_data_policy;
    laplacian<OutOfRangePolicy>(InputNoDataPolicy(), output_no_data_policy,
        execution_policy, value, result);
}


template<
    class ExecutionPolicy,
    class Value,
    class Result
>
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
