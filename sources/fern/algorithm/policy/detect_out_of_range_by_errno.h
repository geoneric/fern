#pragma once
#include <cerrno>


namespace fern {

template<
    class... Parameters>
class DetectOutOfRangeByErrno
{

public:

    static constexpr bool
                   within_range        (Parameters const&... parameters);

};


template<
    class... Parameters>
inline constexpr bool DetectOutOfRangeByErrno<Parameters...>::within_range(
    Parameters const&... /* parameters */)
{
    return errno != ERANGE;
}

} // namespace fern
