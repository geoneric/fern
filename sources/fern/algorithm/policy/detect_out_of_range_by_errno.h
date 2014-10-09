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
