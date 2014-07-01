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

//! Convolve \a source by \a kernel and write results to \a destination.
/*!
  \param[in] source Image to read values from to convolve.
  \param[in] kernel Kernel containing the weights to use.
  \param[out] destination Image to write calculated values to.
*/
template<
    class AlternativeForNoDataPolicy,
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
        AlternativeForNoDataPolicy,
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
    class AlternativeForNoDataPolicy,
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
    convolve<AlternativeForNoDataPolicy, NormalizePolicy, OutOfImagePolicy,
        OutOfRangePolicy>(
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
    using AlternativeForNoDataPolicy=convolve::SkipNoData;
    using NormalizePolicy=convolve::DivideByWeights;
    using OutOfImagePolicy=convolve::SkipOutOfImage;
    using InputNoDataPolicy=SkipNoData;
    using OutputNoDataPolicy=DontMarkNoData;

    convolve<AlternativeForNoDataPolicy, NormalizePolicy, OutOfImagePolicy,
        nullary::DiscardRangeErrors>(
            InputNoDataPolicy(), OutputNoDataPolicy(),
            std::forward<ExecutionPolicy>(execution_policy),
            source, kernel, destination);
}

} // namespace convolution
} // namespace fern
