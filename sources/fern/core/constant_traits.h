#pragma once
#include "fern/core/argument_traits.h"


namespace fern {

// Declarations of functions that are used in the implementation of operations.
// These are not defined. For each constant type they need to be implemented.
// See also argument_traits.h, masked_array_traits.h, ...

template<
    class T>
T const&           get                 (T const& constant);

template<
    class T>
T&                 get                 (T& constant);


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
