#pragma once
#include <cstdint>
#include "fern/core/data_traits.h"


namespace fern {

#define CONSTANT_DATA_TRAITS(                               \
    T)                                                      \
template<>                                                  \
struct DataTraits<T>                                        \
{                                                           \
                                                            \
    using argument_category = constant_tag;                 \
                                                            \
    template<                                               \
        class U>                                            \
    struct Clone                                            \
    {                                                       \
        using type = U;                                     \
    };                                                      \
                                                            \
    using value_type = T;                                   \
                                                            \
    using reference = T&;                                   \
                                                            \
    using const_reference = T const&;                       \
                                                            \
    static bool const is_masking = false;                   \
                                                            \
    static size_t const rank = 0u;                          \
                                                            \
};


CONSTANT_DATA_TRAITS(bool)
CONSTANT_DATA_TRAITS(uint8_t)
CONSTANT_DATA_TRAITS(uint16_t)
CONSTANT_DATA_TRAITS(uint32_t)
CONSTANT_DATA_TRAITS(uint64_t)
CONSTANT_DATA_TRAITS(int8_t)
CONSTANT_DATA_TRAITS(int16_t)
CONSTANT_DATA_TRAITS(int32_t)
CONSTANT_DATA_TRAITS(int64_t)
CONSTANT_DATA_TRAITS(float)
CONSTANT_DATA_TRAITS(double)

#undef CONSTANT_DATA_TRAITS

} // namespace fern
