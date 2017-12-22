// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/core/base_class.h"
#include "fern/core/math.h"
#include "fern/algorithm/policy/policies.h"
#include "fern/algorithm/algebra/elementary/detail/pow.h"


namespace fern {
namespace algorithm {
namespace pow {

/*!
    @ingroup    fern_algorithm_algebra_elementary_group
    @brief      Out of domain policy for fern::algorithm::algebra::pow
                algorithm.
    @sa         @ref fern_algorithm_policies_out_of_domain_policy

    In the folowing cases, within_domain(Base const&, Exponent const&)
    returns false:
    - @a base < 0 and @a base's exponent is not 0.
    - @a base == 0 and @a exponent < 0.

    The value types of @a base and @a exponent must be floating point and
    the same.
*/
template<
    typename Base,
    typename Exponent>
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

            if(is_not_equal(fractional, Base(0))) {
                return false;
            }
        }
        else if(is_equal(base, Base(0)) && exponent < Exponent(0)) {
            return false;
        }

        return true;
    }

};


/*!
    @ingroup    fern_algorithm_algebra_elementary_group
    @brief      Out of range policy for fern::algorithm::algebra::pow algorithm.
    @sa         fern::algorithm::DetectOutOfRangeByErrno,
                @ref fern_algorithm_policies_out_of_range_policy

    The value types of @a base and @a exponent must be floating point and
    the same.
*/
template<
    typename Base,
    typename Exponent,
    typename Result>
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

/*!
    @ingroup    fern_algorithm_algebra_elementary_group
    @brief      Raise @a base to the power of @a exponent and write the
                result to @a result.
    @sa         fern::algorithm::pow::OutOfDomainPolicy,
                fern::algorithm::pow::OutOfRangePolicy,
                fern::algorithm::binary_local_operation

    The value types of @a base, @a exponent and @a result must be floating
    point and the same.
*/
template<
    template<typename, typename> class OutOfDomainPolicy,
    template<typename, typename, typename> class OutOfRangePolicy,
    typename InputNoDataPolicy,
    typename OutputNoDataPolicy,
    typename ExecutionPolicy,
    typename Base,
    typename Exponent,
    typename Result
>
void pow(
    InputNoDataPolicy const& input_no_data_policy,
    OutputNoDataPolicy& output_no_data_policy,
    ExecutionPolicy& execution_policy,
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
    @ingroup    fern_algorithm_algebra_elementary_group
    @overload
*/
template<
    typename ExecutionPolicy,
    typename Base,
    typename Exponent,
    typename Result
>
void pow(
    ExecutionPolicy& execution_policy,
    Base const& base,
    Exponent const& exponent,
    Result& result)
{
    using InputNoDataPolicy = InputNoDataPolicies<SkipNoData, SkipNoData>;
    using OutputNoDataPolicy = DontMarkNoData;

    OutputNoDataPolicy output_no_data_policy;
    pow<binary::DiscardDomainErrors, binary::DiscardRangeErrors>(
        InputNoDataPolicy{{}, {}}, output_no_data_policy, execution_policy,
        base, exponent, result);
}

} // namespace algebra
} // namespace algorithm
} // namespace fern
