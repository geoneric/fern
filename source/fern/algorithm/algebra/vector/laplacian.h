#pragma once
#include "fern/core/assert.h"
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/algebra/vector/detail/laplacian.h"


namespace fern {
namespace algorithm {
namespace laplacian {

/*!
    @ingroup    fern_algorithm_algebra_vector_group
    @brief      The out-of-range policy for the laplacian operation.

    The result of the laplacian operation is a floating point. This policy
    verifies whether the result value is finite.
*/
template<
    typename Value,
    typename Result>
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

/*!
    @ingroup    fern_algorithm_algebra_vector_group
    @brief      Calculate the laplacian of @a value and write the result to
                @a result.
    @sa         fern::algorithm::laplacian::OutOfRangePolicy

    The algorithm implemented is similar to [the one implemented in PCRaster]
    (https://sourceforge.net/p/pcraster/pcrtree2/ci/master/tree/sources/calc/vf.c).

    In short/pseudo code, the algorithm:

    @code
    convolve(value, kernel, result);
    result = (result - (sum(kernel) * value)) / cell_area;
    result = ((convolve(value, kernel) -
        (convolve(defined(value), kernel(1)) * value)) / cell_area;
    @endcode

    Kernel:

    @code
    +---+---+---+
    | 2 | 3 | 2 |
    +---+---+---+
    | 3 | 0 | 3 |
    +---+---+---+
    | 2 | 3 | 2 |
    +---+---+---+
    @endcode

    The value type of @a value and @a result must be floating point and the
    same.
*/
template<
    template<typename, typename> class OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename ExecutionPolicy,
    typename Value,
    typename Result>
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


/*!
    @ingroup    fern_algorithm_algebra_vector_group
    @overload
*/
template<
    typename ExecutionPolicy,
    typename Value,
    typename Result>
void laplacian(
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    using InputNoDataPolicy = InputNoDataPolicies<SkipNoData<>>;
    using OutputNoDataPolicy = DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;
    laplacian<unary::DiscardRangeErrors>(InputNoDataPolicy{{}},
        output_no_data_policy, execution_policy, value, result);
}

} // namespace algebra
} // namespace algorithm
} // namespace fern
