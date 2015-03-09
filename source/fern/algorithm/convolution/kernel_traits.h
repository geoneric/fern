#pragma once
#include <type_traits>
#include "fern/core/data_traits.h"


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
