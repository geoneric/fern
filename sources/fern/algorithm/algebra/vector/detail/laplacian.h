#pragma once
#include <utility>
#include "fern/core/assert.h"
#include "fern/core/value_type.h"
#include "fern/feature/core/array_traits.h"
#include "fern/feature/core/masked_array.h"
#include "fern/feature/core/masked_raster.h"
#include "fern/algorithm/core/cast.h"
#include "fern/algorithm/convolution/convolve.h"
#include "fern/algorithm/convolution/dont_divide_by_weights.h"
#include "fern/algorithm/convolution/neighborhood/square_traits.h"
#include "fern/algorithm/algebra/elementary/divide.h"
#include "fern/algorithm/algebra/elementary/multiply.h"
#include "fern/algorithm/algebra/elementary/subtract.h"
#include "fern/algorithm/algebra/boolean/not.h"
#include "fern/algorithm/statistic/sum.h"


namespace fern {
namespace laplacian {
namespace detail {
namespace dispatch {

template<class Values, class Result,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class RasterCollectionCategory>
class Laplacian
{
};


template<class Values, class Result,
    class InputNoDataPolicy,
    class OutputNoDataPolicy>
class Laplacian<Values, Result,
        InputNoDataPolicy,
        OutputNoDataPolicy,
        raster_2d_tag>:

    public InputNoDataPolicy,
    public OutputNoDataPolicy

{

public:

    Laplacian()
        : InputNoDataPolicy(),
          OutputNoDataPolicy()
    {
    }

    Laplacian(
        InputNoDataPolicy&& input_no_data_policy,
        OutputNoDataPolicy&& output_no_data_policy)
        : InputNoDataPolicy(std::forward<InputNoDataPolicy>(
              input_no_data_policy)),
          OutputNoDataPolicy(std::forward<OutputNoDataPolicy>(
              output_no_data_policy))
    {
    }

    // 2d raster
    inline void calculate(
        Values const& values,
        Result& result)
    {
        using Float = value_type<Values>;
        FERN_STATIC_ASSERT(std::is_floating_point, Float)

        // Convolve using the kernel below, without normalizing the result
        // by the kernel weights.
        Square<Float, 1> kernel({
            {2, 3, 2},
            {3, 0, 3},
            {2, 3, 2}
        });

        convolution::convolve<
            convolve::SkipNoData,
            convolve::DontDivideByWeights,
            convolve::SkipOutOfImage,
            // TODO: How to select this one based on the policy passed into
            //       laplacian?
            convolve::OutOfRangePolicy>(
                static_cast<InputNoDataPolicy&>(*this),
                static_cast<OutputNoDataPolicy&>(*this),
                fern::sequential,  // TODO Use policy passed into laplacian.
                values, kernel, result);

        // Calculate the sum of the kernel weights. This equals the convolution
        // of the mask (inverted and converted to integers), by the kernel
        // used above, and without normalizing by the kernel weights.
        auto extents = fern::extents[size(values, 0)][size(values, 1)];

        // Determine which cells have a valid value.
        Array<bool, 2> inverted_mask(extents);
        algebra::not_(
            static_cast<InputNoDataPolicy&>(*this),
            static_cast<OutputNoDataPolicy&>(*this),
            fern::sequential,  // TODO Use policy passed into laplacian.
            values.mask(), inverted_mask);

        // Cast array of bool to array of floats.
        Array<Float, 2> inverted_mask_as_floats(extents);
        // TODO Out of range handling depends on output not data policy.
        core::cast<cast::OutOfRangePolicy>(
            static_cast<InputNoDataPolicy&>(*this),
            static_cast<OutputNoDataPolicy&>(*this),
            fern::sequential,  // TODO Use policy passed into laplacian.
            inverted_mask, inverted_mask_as_floats);

        Array<Float, 2> sum_of_weights(extents);
        convolution::convolve<
            convolve::SkipNoData,
            convolve::DontDivideByWeights,
            convolve::SkipOutOfImage,
            // TODO: How to select this one based on the policy passed into
            //       laplacian?
            convolve::OutOfRangePolicy>(
                static_cast<InputNoDataPolicy&>(*this),
                static_cast<OutputNoDataPolicy&>(*this),
                fern::sequential,  // TODO Use policy passed into laplacian.
                inverted_mask_as_floats, kernel, sum_of_weights);

        // Multiply the values by the sum of weights.
        MaskedRaster<Float, 2> multiplied_values(extents,
            result.transformation());
        algebra::multiply<
            fern::multiply::OutOfRangePolicy>(  // TODO Pick correct policy.
            static_cast<InputNoDataPolicy&>(*this),
            static_cast<OutputNoDataPolicy&>(*this),
            fern::sequential,  // TODO Use policy passed into laplacian.
            sum_of_weights, values, multiplied_values);

        // Subtract the convolution result by the multiplied values.
        // Result subtracted_results;
        algebra::subtract<
            fern::subtract::OutOfRangePolicy>(  // TODO Pick correct policy.
            static_cast<InputNoDataPolicy&>(*this),
            static_cast<OutputNoDataPolicy&>(*this),
            fern::sequential,  // TODO Use policy passed into laplacian.
            result, multiplied_values, result);

        // Divide subtracted results by the area of the cells.
        algebra::divide<
            fern::divide::OutOfDomainPolicy,  // TODO Pick correct policy.
            fern::divide::OutOfRangePolicy>(  // TODO Pick correct policy.
            static_cast<InputNoDataPolicy&>(*this),
            static_cast<OutputNoDataPolicy&>(*this),
            fern::sequential,  // TODO Use policy passed into laplacian.
            result, cell_area(values), result);
    }

};

} // namespace dispatch
} // namespace detail
} // namespace laplacian
} // namespace fern
