#pragma once
#include <utility>
#include "fern/core/assert.h"
#include "fern/core/value_type.h"
#include "fern/core/argument_categories.h"
#include "fern/core/argument_traits.h"
#include "fern/core/assert.h"
// #include "fern/core/collection_traits.h"
#include "fern/feature/core/array_traits.h"
#include "fern/feature/core/masked_array_traits.h"
#include "fern/algorithm/convolution/neighborhood/square_traits.h"
#include "fern/algorithm/core/index_ranges.h"
#include "fern/algorithm/algebra/elementary/add.h"
#include "fern/algorithm/algebra/elementary/divide.h"
#include "fern/algorithm/algebra/elementary/multiply.h"
#include "fern/algorithm/algebra/elementary/pow.h"
#include "fern/algorithm/algebra/elementary/sqrt.h"
#include "fern/algorithm/convolution/convolve.h"
#include "fern/algorithm/convolution/policies.h"


namespace fern {
namespace slope {
namespace detail {
namespace dispatch {

template<class Values, class Result,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ArgumentCollectionCategory>
class Slope
{
};


template<class Values, class Result,
    class InputNoDataPolicy,
    class OutputNoDataPolicy>
class Slope<Values, Result,
        InputNoDataPolicy,
        OutputNoDataPolicy,
        raster_2d_tag>:

    public InputNoDataPolicy,
    public OutputNoDataPolicy

{

    FERN_STATIC_ASSERT(std::is_floating_point, value_type<Values>)
    FERN_STATIC_ASSERT(std::is_same, value_type<Values>, value_type<Result>)

public:

    Slope()
        : InputNoDataPolicy(),
          OutputNoDataPolicy()
    {
    }

    Slope(
        InputNoDataPolicy&& input_no_data_policy,
        OutputNoDataPolicy&& output_no_data_policy)
        : InputNoDataPolicy(std::forward<InputNoDataPolicy>(
              input_no_data_policy)),
          OutputNoDataPolicy(std::forward<OutputNoDataPolicy>(
              output_no_data_policy))
    {
    }

    // 2d array
    inline void calculate(
        Values const& values,
        Result& result)
    {
        size_t const size1 = fern::size(values, 0);
        size_t const size2 = fern::size(values, 1);

        auto ranges = IndexRanges<2>{
            IndexRange(0, size1),
            IndexRange(0, size2)
        };

        calculate(ranges, values, result);
    }

    template<
        class Indices>
    inline void calculate(
        Indices const& /* indices */,
        Values const& values,
        Result& result)
    {
        using Float = value_type<Values>;
        FERN_STATIC_ASSERT(std::is_floating_point, Float)

        // dz_dx: convolve using this kernel.
        Square<Float, 1> dz_dx_kernel({
            {1, 0, -1},
            {2, 0, -2},
            {1, 0, -1}
        });

        // dz_dx: convolve using this kernel.
        Square<Float, 1> dz_dy_kernel({
            {-1, -2, -1},
            {0, 0, 0},
            {1, 2, 1}
        });

        auto extents = fern::extents[size(values, 0)][size(values, 1)];

        MaskedArray<Float, 2> dz_dx(extents);
        convolution::convolve<
            convolve::ReplaceNoDataByFocalAverage,
            convolve::DontDivideByWeights,
            convolve::ReplaceOutOfImageByFocalAverage,
            convolve::OutOfRangePolicy>(
                static_cast<InputNoDataPolicy&>(*this),
                static_cast<OutputNoDataPolicy&>(*this),
                sequential,  // TODO Depends on exec policy passed into slope.
                values, dz_dx_kernel, dz_dx);

        algebra::divide(dz_dx, 8 * cell_size(values, 0), dz_dx);

        MaskedArray<Float, 2> dz_dy(extents);

        convolution::convolve<
            convolve::ReplaceNoDataByFocalAverage,
            convolve::DontDivideByWeights,
            convolve::ReplaceOutOfImageByFocalAverage,
            convolve::OutOfRangePolicy>(
                static_cast<InputNoDataPolicy&>(*this),
                static_cast<OutputNoDataPolicy&>(*this),
                fern::sequential, values, dz_dy_kernel, dz_dy);

        algebra::divide(dz_dy, 8 * cell_size(values, 1), dz_dy);

        algebra::pow(dz_dx, Float(2), dz_dx);
        algebra::pow(dz_dy, Float(2), dz_dy);
        algebra::add(dz_dx, dz_dy, result);
        /// algebra::sqrt(result, result);

        // TODO Whether or not to detect out of domain values depends on
        //      output no-data policy passed in. Will sqrt succeed if input
        //      < 0?
        algebra::sqrt<sqrt::OutOfDomainPolicy>(
            static_cast<InputNoDataPolicy&>(*this),
            static_cast<OutputNoDataPolicy&>(*this),
            sequential,  // TODO Depends on exec policy passed into slope.
            result, result);
    }

};

} // namespace dispatch
} // namespace detail
} // namespace slope
} // namespace fern
