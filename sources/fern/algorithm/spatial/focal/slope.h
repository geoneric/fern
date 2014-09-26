#pragma once
#include "fern/core/assert.h"
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/spatial/focal/detail/slope.h"


namespace fern {
namespace slope {

//! The out-of-range policy for the slope operation.
/*!
  The result of the slope operation is a floating point. This policy
  verifies whether the result value is finite.
*/
template<
    class Value,
    class Result>
class OutOfRangePolicy
{

    FERN_STATIC_ASSERT(std::is_floating_point, value_type<Value>)
    FERN_STATIC_ASSERT(std::is_same, value_type<Result>, value_type<Value>)

public:

    inline bool within_range(
        Result const& result) const
    {
        return std::isfinite(result);
    }

};

} // namespace slope


namespace spatial {

//! Calculate the slope of \a value and write the result to \a result.
/*!
    \ingroup       spatial
    \sa            fern::slope::OutOfRangePolicy,
                   @ref fern_algorithm_spatial

    This algorithm implements Horne's slope algorithm (Horn, B.K.P. (1981)
    Hill shading and the reflectance map. Proceedings of IEEE 69(1), 14-47).
    In pseudo-code this works as folows:

    \code
    dz_dx = convolve(value, dz_dx_kernel) / (8 * cell_size)
    dz_dy = convolve(value, dz_dy_kernel) / (8 * cell_size)
    result = sqrt(pow(dz_dx, 2) + pow(dz_dy, 2))
    \endcode

    where dz_dx_kernel is:

    \code
    +----+----+----+
    |  1 |  0 | -1 |
    +----+----+----+
    |  2 |  0 | -2 |
    +----+----+----+
    |  1 |  0 | -1 |
    +----+----+----+
    \endcode

    and dz_dy_kernel is:

    \code
    +----+----+----+
    | -1 | -2 | -1 |
    +----+----+----+
    |  0 |  0 |  0 |
    +----+----+----+
    |  1 |  2 |  1 |
    +----+----+----+
    \endcode
*/
template<
    template<class, class> class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Value,
    class Result
>
void slope(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    FERN_STATIC_ASSERT(std::is_floating_point, value_type<Value>)
    FERN_STATIC_ASSERT(std::is_same, value_type<Result>, value_type<Value>)

    slope::detail::slope<OutOfRangePolicy>(input_no_data_policy,
        output_no_data_policy, execution_policy, value, result);
}


/*!
    \ingroup       spatial
    \overload
*/
template<
    template<class, class> class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Value,
    class Result
>
void slope(
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    OutputNoDataPolicy output_no_data_policy;
    slope<OutOfRangePolicy>(InputNoDataPolicy(), output_no_data_policy,
        execution_policy, value, result);
}


/*!
    \ingroup       spatial
    \overload
*/
template<
    class ExecutionPolicy,
    class Value,
    class Result
>
void slope(
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    using InputNoDataPolicy = SkipNoData<>;
    using OutputNoDataPolicy = DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;
    slope<unary::DiscardRangeErrors>(InputNoDataPolicy(),
        output_no_data_policy, execution_policy, value, result);
}

} // namespace spatial
} // namespace fern
