#pragma once
#include <utility>
#include "fern/core/assert.h"
#include "fern/core/thread_client.h"
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

template<
    class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value,
    class Result,
    class ExecutionPolicy,
    class ValueCollectionCategory>
class Laplacian
{
};


/// template<
///     template<class, class> class Template1,
///     template<class, class> class Template2,
///     bool condition
/// >
/// struct SelectIf
/// {
/// };
/// 
/// 
/// template<
///     template<class, class> class Template1,
///     template<class, class> class Template2
/// >
/// struct SelectIf<
///     Template1,
///     Template2,
///     true>
/// {
///     template<class V1, class V2>
///     using Template = Template1<V1, V2>;
/// };
/// 
/// 
/// template<
///     template<class, class> class Template1,
///     template<class, class> class Template2
/// >
/// struct SelectIf<
///     Template1,
///     Template2,
///     false>
/// {
///     template<class V1, class V2>
///     using Template = Template2<V1, V2>;
/// };
/// 
/// 
/// template<class Value, class Result>
/// using OutOfRangePolicy1 = SelectIf<convolve::OutOfRangePolicy, convolve::OutOfRangePolicy, true>::Template<Value, Result>;


// typename SelectIf<
//     unary::DiscardRangeErrors,
//     convolve::OutOfRangePolicy,
//     std::is_same<OutputNoDataPolicy, DontMarkNoData>::value>
//     .template ::Template Bla;


template<
    class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class Value,
    class Result,
    class ExecutionPolicy>
struct Laplacian<
    OutOfRangePolicy,
    InputNoDataPolicy,
    OutputNoDataPolicy,
    Value,
    Result,
    ExecutionPolicy,
    array_2d_tag>

{

    // f(2d array)
    static void apply(
        InputNoDataPolicy const& input_no_data_policy,
        OutputNoDataPolicy& output_no_data_policy,
        ExecutionPolicy const& execution_policy,
        Value const& value,
        Result& result)
    {
        assert(fern::size(value, 0) == fern::size(result, 0));
        assert(fern::size(value, 1) == fern::size(result, 1));

        using Float = value_type<Value>;
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
            // TODO: Select OutOfRange policy based on the
            //       output-no-data-policy passed in.
            convolve::OutOfRangePolicy>(
                input_no_data_policy, output_no_data_policy, execution_policy,
                value, kernel, result);

        // Calculate the sum of the kernel weights. This equals the convolution
        // of the mask (inverted and converted to integers), by the kernel
        // used above, and without normalizing by the kernel weights.
        auto extents = fern::extents[size(value, 0)][size(value, 1)];

        // Determine which cells have a valid value.
        Array<bool, 2> inverted_mask(extents);
        algebra::not_(
            input_no_data_policy, output_no_data_policy, execution_policy,
            value.mask(), inverted_mask);

        // Cast array of bool to array of floats.
        Array<Float, 2> inverted_mask_as_floats(extents);
        // TODO: Select OutOfRange policy based on the
        //       output-no-data-policy passed in.
        core::cast<cast::OutOfRangePolicy>(
            input_no_data_policy, output_no_data_policy, execution_policy,
            inverted_mask, inverted_mask_as_floats);

        Array<Float, 2> sum_of_weights(extents);
        convolution::convolve<
            convolve::SkipNoData,
            convolve::DontDivideByWeights,
            convolve::SkipOutOfImage,
            // TODO: Select OutOfRange policy based on the
            //       output-no-data-policy passed in.
            convolve::OutOfRangePolicy>(
                input_no_data_policy, output_no_data_policy, execution_policy,
                inverted_mask_as_floats, kernel, sum_of_weights);

        // Multiply the values by the sum of weights.
        MaskedRaster<Float, 2> multiplied_values(extents,
            result.transformation());
        algebra::multiply<
            // TODO: Select OutOfRange policy based on the
            //       output-no-data-policy passed in.
            fern::multiply::OutOfRangePolicy>(
                input_no_data_policy, output_no_data_policy, execution_policy,
                sum_of_weights, value, multiplied_values);

        // Subtract the convolution result by the multiplied values.
        // Result subtracted_results;
        algebra::subtract<
            // TODO: Select OutOfRange policy based on the
            //       output-no-data-policy passed in.
            fern::subtract::OutOfRangePolicy>(
                input_no_data_policy, output_no_data_policy, execution_policy,
                result, multiplied_values, result);

        // Divide subtracted results by the area of the cells.
        algebra::divide<
            // TODO: Select OutOfDomain policy based on the
            //       output-no-data-policy passed in.
            // TODO: Select OutOfRange policy based on the
            //       output-no-data-policy passed in.
            fern::divide::OutOfDomainPolicy,
            fern::divide::OutOfRangePolicy>(
                input_no_data_policy, output_no_data_policy, execution_policy,
                result, cell_area(value), result);
    }

};

} // namespace dispatch


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
    dispatch::Laplacian<
        OutOfRangePolicy<value_type<Value>, value_type<Result>>,
        InputNoDataPolicy,
        OutputNoDataPolicy,
        Value,
        Result,
        ExecutionPolicy,
        base_class<argument_category<Value>, array_2d_tag>>::apply(
            input_no_data_policy, output_no_data_policy, execution_policy,
            value, result);
}

} // namespace detail
} // namespace laplacian
} // namespace fern
