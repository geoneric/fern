#pragma once
#include "fern/core/assert.h"
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/algebra/elementary/detail/sqrt.h"


namespace fern {
namespace algorithm {
namespace sqrt {

//! Out of domain policy for fern::algebra::sqrt algorithm.
/*!
    Values smaller than 0 are considered out-of-domain.

    The value types of \a value and \a result must be floating point and
    the same.

    \sa            @ref fern_algorithm_policies_out_of_domain_policy
*/
template<
    class Value>
class OutOfDomainPolicy
{

    FERN_STATIC_ASSERT(std::is_floating_point, Value)

public:

    inline static bool within_domain(
        Value const& value)
    {
        return value >= Value(0);
    }

};


} // namespace sqrt


namespace algebra {

//! Calculate the square root of \a value and write the result to \a result.
/*!
    \ingroup       elementary
    \sa            fern::sqrt::OutOfDomainPolicy, fern::unary_local_operation,
                   @ref fern_algorithm_algebra_elementary

    The value types of \a value and \a result must be floating point and the
    same.
*/
template<
    template<class> class OutOfDomainPolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Value,
    class Result
>
void sqrt(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    FERN_STATIC_ASSERT(std::is_floating_point, value_type<Value>)
    FERN_STATIC_ASSERT(std::is_same, value_type<Result>, value_type<Value>)

    sqrt::detail::sqrt<OutOfDomainPolicy>(input_no_data_policy,
        output_no_data_policy, execution_policy, value, result);
}


/*!
    \ingroup       elementary
    \overload
*/
template<
    template<class> class OutOfDomainPolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Value,
    class Result
>
void sqrt(
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    OutputNoDataPolicy output_no_data_policy;
    sqrt<OutOfDomainPolicy>(InputNoDataPolicy(), output_no_data_policy,
        execution_policy, value, result);
}


/*!
    \ingroup       elementary
    \overload
*/
template<
    class ExecutionPolicy,
    class Value,
    class Result
>
void sqrt(
    ExecutionPolicy const& execution_policy,
    Value const& value,
    Result& result)
{
    using InputNoDataPolicy = SkipNoData<>;
    using OutputNoDataPolicy = DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;
    sqrt<unary::DiscardDomainErrors>(InputNoDataPolicy(),
        output_no_data_policy, execution_policy, value, result);
}

} // namespace algebra
} // namespace algorithm
} // namespace fern
