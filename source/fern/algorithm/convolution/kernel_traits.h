// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <type_traits>
#include "fern/core/data_type_traits.h"


namespace fern {
namespace algorithm {

template<
    typename Kernel>
struct KernelTraits
{
    static bool const weigh_values{!std::is_same<value_type<Kernel>, bool>()};
};

} // namespace algorithm
} // namespace fern
