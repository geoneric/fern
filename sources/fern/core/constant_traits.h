#pragma once


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


#define CONSTANT_ARGUMENT_TRAITS(           \
    T)                                      \
template<>                                  \
struct ArgumentTraits<T>                    \
{                                           \
                                            \
    typedef constant_tag argument_category; \
                                            \
    template<                               \
        class U>                            \
    struct Constant                         \
    {                                       \
        typedef U type;                     \
    };                                      \
                                            \
    typedef T value_type;                   \
                                            \
    static bool const is_masking = false;   \
                                            \
};                                          \
                                            \
                                            \
template<>                                  \
inline T const& get(                        \
    T const& constant)                      \
{                                           \
    return constant;                        \
}                                           \
                                            \
                                            \
template<>                                  \
inline T& get(                              \
    T& constant)                            \
{                                           \
    return constant;                        \
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
