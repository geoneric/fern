#pragma once
#include "fern/core/assert.h"
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/convolution/policies.h"
#include "fern/algorithm/convolution/detail/convolve.h"


namespace fern {
namespace convolve {

template<
    class Result>
using OutOfRangePolicy = detail::OutOfRangePolicy<Result>;

} // namespace convolve


namespace convolution {

/// template<
///     // class ExecutionPolicy,
///     class SourceImage,
///     class Kernel,
///     class DestinationImage,
///     class NormalizePolicy=convolve::DivideByWeights,
///     template<class> class OutOfRangePolicy=nullary::DiscardRangeErrors,
///     class InputNoDataPolicy=SkipNoData,
///     class OutputNoDataPolicy=DontMarkNoData
/// >
/// class Convolve
/// {
/// 
/// public:
/// 
///     using category = focal_operation_tag;
///     using A1 = SourceImage;
///     using A1Value = value_type<A1>;
///     using A2 = Kernel;
///     using A2Value = value_type<A2>;
///     using R = DestinationImage;
///     using RValue = value_type<R>;
/// 
///     FERN_STATIC_ASSERT(std::is_arithmetic, A1Value)
///     FERN_STATIC_ASSERT(std::is_arithmetic, A2Value)
///     FERN_STATIC_ASSERT(std::is_arithmetic, RValue)
/// 
///     FERN_STATIC_ASSERT(std::is_floating_point, A1Value)
///     FERN_STATIC_ASSERT(std::is_floating_point, RValue)
/// 
///     Convolve()
///         : _algorithm()
///     {
///     }
/// 
///     Convolve(
///         InputNoDataPolicy&& input_no_data_policy,  // Universal reference.
///         OutputNoDataPolicy&& output_no_data_policy)  // Universal reference.
///         : _algorithm(
///             std::forward<InputNoDataPolicy>(input_no_data_policy),
///             std::forward<OutputNoDataPolicy>(output_no_data_policy))
///     {
///     }
/// 
///     inline void operator()(
///         A1 const& source_image,
///         A2 const& kernel,
///         R& destination_image)
///     {
///         _algorithm.calculate(source_image, kernel, destination_image);
///     }
/// 
///     // template<
///     //     class Indices>
///     // inline void operator()(
///     //     Indices const& indices,
///     //     A const& values,
///     //     R& result)
///     // {
///     //     _algorithm.calculate(indices, values, result);
///     // }
/// 
/// private:
/// 
///     convolve::detail::dispatch::Convolve<A1, A2, R,
///         NormalizePolicy, OutOfRangePolicy,
///         InputNoDataPolicy, OutputNoDataPolicy,
///         typename base_class<
///             typename ArgumentTraits<A1>::argument_category,
///             array_2d_tag>::type,
///         typename base_class<
///             typename ArgumentTraits<A2>::argument_category,
///             array_2d_tag>::type,
///         typename base_class<
///             typename ArgumentTraits<R>::argument_category,
///             array_2d_tag>::type> _algorithm;
/// 
/// };


//! Convolve \a source by \a kernel and write results to \a destination.
/*!
  \param[in] source Image to read values from to convolve.
  \param[in] kernel Kernel containing the weights to use.
  \param[out] destination Image to write calculated values to.
*/
template<
    class NormalizePolicy,
    class OutOfImagePolicy,
    template<class> class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class SourceImage,
    class Kernel,
    class DestinationImage
>
void convolve(
    InputNoDataPolicy&& input_no_data_policy,
    OutputNoDataPolicy&& output_no_data_policy,
    ExecutionPolicy&& execution_policy,
    SourceImage const& source,
    Kernel const& kernel,
    DestinationImage& destination)
{
    using SourceValue = value_type<SourceImage>;
    using KernelValue = value_type<Kernel>;
    using DestinationValue = value_type<DestinationImage>;

    FERN_STATIC_ASSERT(std::is_arithmetic, SourceValue)
    FERN_STATIC_ASSERT(std::is_arithmetic, KernelValue)
    FERN_STATIC_ASSERT(std::is_arithmetic, DestinationValue)

    FERN_STATIC_ASSERT(std::is_floating_point, SourceValue)
    FERN_STATIC_ASSERT(std::is_floating_point, DestinationValue)

    convolve::detail::convolve<
        NormalizePolicy,
        OutOfImagePolicy,
        OutOfRangePolicy>(
            std::forward<InputNoDataPolicy>(input_no_data_policy),
            std::forward<OutputNoDataPolicy>(output_no_data_policy),
            std::forward<ExecutionPolicy>(execution_policy),
            source, kernel, destination);
}


/*!
    \overload

    Use this overload to for passing the types of non-default policies.

    \todo Mention default policies.
*/
template<
    class NormalizePolicy,
    class OutOfImagePolicy,
    template<class> class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class SourceImage,
    class Kernel,
    class DestinationImage
>
void convolve(
    ExecutionPolicy&& execution_policy,
    SourceImage const& source,
    Kernel const& kernel,
    DestinationImage& destination)
{
    convolve<NormalizePolicy, OutOfImagePolicy,
        OutOfRangePolicy /* , InputNoDataPolicy, OutputNoDataPolicy */>(
            InputNoDataPolicy(), OutputNoDataPolicy(),
            std::forward<ExecutionPolicy>(execution_policy),
            source, kernel, destination);
}


/*!
    \overload

    Use this overload if the default policies are fine.

    \todo Mention default policies.
*/
template<
    class ExecutionPolicy,
    class SourceImage,
    class Kernel,
    class DestinationImage
>
void convolve(
    ExecutionPolicy&& execution_policy,
    SourceImage const& source,
    Kernel const& kernel,
    DestinationImage& destination)
{
    using NormalizePolicy=convolve::DivideByWeights;
    using OutOfImagePolicy=convolve::SkipOutOfImage;
    using InputNoDataPolicy=SkipNoData;
    using OutputNoDataPolicy=DontMarkNoData;

    convolve<NormalizePolicy, OutOfImagePolicy,
        nullary::DiscardRangeErrors /* , ExecutionPolicy, InputNoDataPolicy,
        OutputNoDataPolicy */>(
            InputNoDataPolicy(), OutputNoDataPolicy(),
            std::forward<ExecutionPolicy>(execution_policy),
            source, kernel, destination);
}

} // namespace convolution
} // namespace fern
