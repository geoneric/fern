// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <cstdint>
#include "fern/core/data_type_traits.h"


namespace fern {

#define CONSTANT_DATA_TYPE_TRAITS(         \
    T)                                     \
template<>                                 \
struct DataTypeTraits<T>                   \
{                                          \
                                           \
    using argument_category = scalar_tag;  \
                                           \
    template<                              \
        class U>                           \
    struct Clone                           \
    {                                      \
        using type = U;                    \
    };                                     \
                                           \
    using value_type = T;                  \
                                           \
    using reference = T&;                  \
                                           \
    using const_reference = T const&;      \
                                           \
    static bool const is_masking = false;  \
                                           \
    static size_t const rank = 0u;         \
                                           \
};


CONSTANT_DATA_TYPE_TRAITS(bool)
CONSTANT_DATA_TYPE_TRAITS(uint8_t)
CONSTANT_DATA_TYPE_TRAITS(uint16_t)
CONSTANT_DATA_TYPE_TRAITS(uint32_t)
CONSTANT_DATA_TYPE_TRAITS(uint64_t)
CONSTANT_DATA_TYPE_TRAITS(int8_t)
CONSTANT_DATA_TYPE_TRAITS(int16_t)
CONSTANT_DATA_TYPE_TRAITS(int32_t)
CONSTANT_DATA_TYPE_TRAITS(int64_t)
CONSTANT_DATA_TYPE_TRAITS(float)
CONSTANT_DATA_TYPE_TRAITS(double)

#undef CONSTANT_DATA_TYPE_TRAITS

} // namespace fern
