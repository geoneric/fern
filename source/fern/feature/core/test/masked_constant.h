// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
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
