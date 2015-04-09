// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/core/parameter_pack.h"


namespace fern {
namespace algorithm {

/*!
    @ingroup    fern_algorithm_policy_group
*/
template<
    typename... Parameters>
class DiscardRangeErrors
{

public:

    //! Type of result parameter.
    using LastParameter = typename last_type<Parameters...>::type;

    static constexpr bool
                   within_range        (LastParameter const& parameter);

    static constexpr bool
                   within_range        (Parameters const&... parameters);

};


template<
    typename... Parameters>
inline constexpr bool DiscardRangeErrors<Parameters...>::within_range(
    LastParameter const& /* parameter */)
{
    return true;
}


template<
    typename... Parameters>
inline constexpr bool DiscardRangeErrors<Parameters...>::within_range(
    Parameters const&... /* parameters */)
{
    return true;
}


namespace nullary {

template<
    typename Result>
using DiscardRangeErrors = DiscardRangeErrors<Result>;

} // namespace nullary


namespace unary {

template<
    typename Value,
    typename Result>
using DiscardRangeErrors = DiscardRangeErrors<Value, Result>;

} // namespace unary


namespace binary {

template<
    typename Value1,
    typename Value2,
    typename Result>
using DiscardRangeErrors = DiscardRangeErrors<Value1, Value2, Result>;

} // namespace binary
} // namespace algorithm
} // namespace fern
