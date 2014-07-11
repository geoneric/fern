#pragma once
#include "fern/core/assert.h"
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/convolution/policies.h"
#include "fern/algorithm/convolution/detail/convolve.h"


namespace fern {
namespace convolve {

//! Out-of-range policy for fern::convolution::convolve algorithm.
/*!
    The result of the convolution operation is a floating point. This policy
    verifies whether the result value is finite.

    \a Value and \a Result must be floating point.

    \sa            @ref fern_algorithm_policies_out_of_range_policy
*/
template<
    class Value,
    class Result>
using OutOfRangePolicy = detail::OutOfRangePolicy<Value, Result>;

} // namespace convolve


namespace convolution {

//! Convolve \a source by \a kernel and write the result to \a destination.
/*!
    \param[in] source Image to read values from to convolve.
    \param[in] kernel Kernel containing the weights to use.
    \param[out] destination Image to write calculated values to.
    \sa         fern::convolve::OutOfRangePolicy

    This is a very flexible algorithm. It is written in terms of a number of
    [policies that handle configurable aspects of convolution]
    (@ref fern_algorithm_convolution_policies).

    The value types of \a source and \a destination must be floating point.
    The value type of \a kernel must be arithmetic.
*/
template<
    class AlternativeForNoDataPolicy,
    class NormalizePolicy,
    class OutOfImagePolicy,
    template<class, class> class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class SourceImage,
    class Kernel,
    class DestinationImage
>
void convolve(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy const& execution_policy,
    SourceImage const& source,
    Kernel const& kernel,
    DestinationImage& destination)
{
    FERN_STATIC_ASSERT(std::is_floating_point, value_type<SourceImage>)
    FERN_STATIC_ASSERT(std::is_arithmetic, value_type<Kernel>)
    FERN_STATIC_ASSERT(std::is_floating_point, value_type<DestinationImage>)

    convolve::detail::convolve<
        AlternativeForNoDataPolicy,
        NormalizePolicy,
        OutOfImagePolicy,
        OutOfRangePolicy>(
            input_no_data_policy, output_no_data_policy,
            execution_policy,
            source, kernel, destination);
}


/*!
    \overload
*/
template<
    class AlternativeForNoDataPolicy,
    class NormalizePolicy,
    class OutOfImagePolicy,
    template<class, class> class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class SourceImage,
    class Kernel,
    class DestinationImage
>
void convolve(
    ExecutionPolicy const& execution_policy,
    SourceImage const& source,
    Kernel const& kernel,
    DestinationImage& destination)
{
    OutputNoDataPolicy output_no_data_policy;

    convolve<AlternativeForNoDataPolicy, NormalizePolicy, OutOfImagePolicy,
        OutOfRangePolicy>(
            InputNoDataPolicy(), output_no_data_policy,
            execution_policy,
            source, kernel, destination);
}


/*!
    \overload

    Use this overload if the default policies are fine. The default policies
    used are:

    Policy                     | Implementation
    -------------------------- | --------------
    AlternativeForNoDataPolicy | fern::convolve::SkipNoData
    NormalizePolicy            | fern::convolve::DivideByWeights
    OutOfImagePolicy           | fern::convolve::SkipOutOfImage
    InputNoDataPolicy          | fern::SkipNoData (as always)
    OutputNoDataPolicy         | fern::DontMarkNoData (as always)
*/
template<
    class ExecutionPolicy,
    class SourceImage,
    class Kernel,
    class DestinationImage
>
void convolve(
    ExecutionPolicy const& execution_policy,
    SourceImage const& source,
    Kernel const& kernel,
    DestinationImage& destination)
{
    using AlternativeForNoDataPolicy = convolve::SkipNoData;
    using NormalizePolicy = convolve::DivideByWeights;
    using OutOfImagePolicy = convolve::SkipOutOfImage;
    using InputNoDataPolicy = SkipNoData;
    using OutputNoDataPolicy = DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;

    convolve<AlternativeForNoDataPolicy, NormalizePolicy, OutOfImagePolicy,
        unary::DiscardRangeErrors>(
            InputNoDataPolicy(), output_no_data_policy,
            execution_policy,
            source, kernel, destination);
}

} // namespace convolution
} // namespace fern
