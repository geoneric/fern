// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <cerrno>


namespace fern {
namespace algorithm {

/*!
    @ingroup    fern_algorithm_policy_group
*/
template<
    typename... Parameters>
class DetectOutOfRangeByErrno
{

public:

    static constexpr bool
                   within_range        (Parameters const&... parameters);

};


template<
    typename... Parameters>
inline constexpr bool DetectOutOfRangeByErrno<Parameters...>::within_range(
    Parameters const&... /* parameters */)
{
    return errno != ERANGE;
}

} // namespace algorithm
} // namespace fern
