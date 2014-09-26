#pragma once
#include "fern/core/base_class.h"
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/algebra/elementary/detail/pow.h"


namespace fern {
namespace pow {

//! Out of domain policy for fern::algebra::pow algorithm.
/*!
    In the folowing cases, within_domain(Base const&, Exponent const&)
    returns false:
    - \a base < 0 and \a base's exponent is not 0.
    - \a base == 0 and \a exponent < 0.

    The value types of \a base and \a exponent must be floating point and
    the same.

    \sa            @ref fern_algorithm_policies_out_of_domain_policy
*/
template<
    class Base,
    class Exponent>
class OutOfDomainPolicy
{

    FERN_STATIC_ASSERT(std::is_floating_point, value_type<Base>)
    FERN_STATIC_ASSERT(std::is_same, value_type<Exponent>, value_type<Base>)

public:

    inline static bool within_domain(
        Base const& base,
        Exponent const& exponent)
    {
        if(base < Base(0)) {
            Base integral, fractional;
            fractional = std::modf(exponent, &integral);

            if(fractional != Base(0)) {
                return false;
            }
        }
        else if(base == Base(0) && exponent < Exponent(0)) {
            return false;
        }

        return true;
    }

};


//! Out of range policy for fern::algebra::pow algorithm.
/*!
    The value types of \a base and \a exponent must be floating point and
    the same.

    \sa            fern::DetectOutOfRangeByErrno,
                   @ref fern_algorithm_policies_out_of_range_policy
*/
template<
    class Base,
    class Exponent,
    class Result>
class OutOfRangePolicy
{

    FERN_STATIC_ASSERT(std::is_floating_point, value_type<Base>)
    FERN_STATIC_ASSERT(std::is_same, value_type<Exponent>, value_type<Base>)
    FERN_STATIC_ASSERT(std::is_same, value_type<Result>, value_type<Base>)

public:

    inline static bool within_range(
        Base const& /* base */,
        Exponent const& /* exponent */,
        Result const& result)
    {
        return std::isfinite(result);
    }

};

} // namespace pow


namespace algebra {

//! Raise \a base to the power of \a exponent and write the result to \a result.
/*!
    \ingroup       elementary
    \sa            fern::pow::OutOfDomainPolicy, fern::pow::OutOfRangePolicy,
                   fern::binary_local_operation,
                   @ref fern_algorithm_algebra_elementary

    The value types of \a base, \a exponent and \a result must be floating
    point and the same.
*/
template<
    template<class, class> class OutOfDomainPolicy,
    template<class, class, class> class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Base,
    class Exponent,
    class Result
>
void pow(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy const& execution_policy,
    Base const& base,
    Exponent const& exponent,
    Result& result)
{
    FERN_STATIC_ASSERT(std::is_floating_point, value_type<Base>)
    FERN_STATIC_ASSERT(std::is_same, value_type<Exponent>, value_type<Base>)
    FERN_STATIC_ASSERT(std::is_same, value_type<Result>, value_type<Base>)

    pow::detail::pow<OutOfDomainPolicy, OutOfRangePolicy>(input_no_data_policy,
        output_no_data_policy, execution_policy, base, exponent, result);
}


/*!
    \ingroup       elementary
    \overload
*/
template<
    template<class, class> class OutOfDomainPolicy,
    template<class, class, class> class OutOfRangePolicy,
    class InputNoDataPolicy,
    class OutputNoDataPolicy,
    class ExecutionPolicy,
    class Base,
    class Exponent,
    class Result
>
void pow(
    ExecutionPolicy const& execution_policy,
    Base const& base,
    Exponent const& exponent,
    Result& result)
{
    OutputNoDataPolicy output_no_data_policy;
    pow<OutOfDomainPolicy, OutOfRangePolicy>(InputNoDataPolicy(),
        output_no_data_policy, execution_policy, base, exponent, result);
}


/*!
    \ingroup       elementary
    \overload
*/
template<
    class ExecutionPolicy,
    class Base,
    class Exponent,
    class Result
>
void pow(
    ExecutionPolicy const& execution_policy,
    Base const& base,
    Exponent const& exponent,
    Result& result)
{
    using InputNoDataPolicy = SkipNoData<>;
    using OutputNoDataPolicy = DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;
    pow<binary::DiscardDomainErrors, binary::DiscardRangeErrors>(
        InputNoDataPolicy(), output_no_data_policy, execution_policy,
        base, exponent, result);
}

} // namespace algebra
} // namespace fern
