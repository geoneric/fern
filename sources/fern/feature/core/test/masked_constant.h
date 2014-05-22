#pragma once
#include <iostream>
#include "fern/feature/core/masked_constant.h"


namespace fern {

template<
    class T>
inline std::ostream& operator<<(
    std::ostream& stream,
    MaskedConstant<T> const& constant)
{
    stream
        << constant.value()
        << '(' << constant.mask() << ')'
        ;
    return stream;
}

} // namespace fern
