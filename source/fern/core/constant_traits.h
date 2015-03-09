#pragma once
#include <cstdint>
#include "fern/core/data_traits.h"
#include "fern/core/collection_traits.h"


namespace fern {

#define CONSTANT_ARGUMENT_TRAITS(                           \
    T)                                                      \
template<>                                                  \
struct DataTraits<T>                                        \
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
};                                                          \
                                                            \
                                                            \
template<>                                                  \
inline typename DataTraits<T>::const_reference get(         \
    typename DataTraits<T>::value_type const& constant)     \
{                                                           \
    return constant;                                        \
}                                                           \
                                                            \
                                                            \
template<>                                                  \
inline typename DataTraits<T>::reference get(               \
    typename DataTraits<T>::value_type& constant)           \
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


template<
    class T>
inline size_t size(
    T const& /* constant */)
{
    return 1u;
}


template<
    class U,
    class V>
inline U clone(
    V const& /* constant */)
{
    return U{};
}


template<
    class U,
    class V>
inline U clone(
    V const& /* value */,
    U const& value)
{
    return U{value};
}

} // namespace fern
