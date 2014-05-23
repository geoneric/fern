#pragma once
#include <cstdint>
#include "fern/core/argument_traits.h"
#include "fern/core/collection_traits.h"


namespace fern {

#define CONSTANT_ARGUMENT_TRAITS(                           \
    T)                                                      \
template<>                                                  \
struct ArgumentTraits<T>                                    \
{                                                           \
                                                            \
    using argument_category = constant_tag;                 \
                                                            \
    template<                                               \
        class U>                                            \
    struct Constant                                         \
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
};                                                          \
                                                            \
                                                            \
template<>                                                  \
inline typename ArgumentTraits<T>::const_reference get(     \
    typename ArgumentTraits<T>::value_type const& constant) \
{                                                           \
    return constant;                                        \
}                                                           \
                                                            \
                                                            \
template<>                                                  \
inline typename ArgumentTraits<T>::reference get(           \
    typename ArgumentTraits<T>::value_type& constant)       \
{                                                           \
    return constant;                                        \
}


CONSTANT_ARGUMENT_TRAITS(bool)
CONSTANT_ARGUMENT_TRAITS(uint8_t)
CONSTANT_ARGUMENT_TRAITS(uint16_t)
CONSTANT_ARGUMENT_TRAITS(uint32_t)
CONSTANT_ARGUMENT_TRAITS(uint64_t)
CONSTANT_ARGUMENT_TRAITS(int8_t)
CONSTANT_ARGUMENT_TRAITS(int16_t)
CONSTANT_ARGUMENT_TRAITS(int32_t)
CONSTANT_ARGUMENT_TRAITS(int64_t)
CONSTANT_ARGUMENT_TRAITS(float)
CONSTANT_ARGUMENT_TRAITS(double)

#undef CONSTANT_ARGUMENT_TRAITS

} // namespace fern
