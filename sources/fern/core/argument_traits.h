#pragma once


namespace fern {

// Argument categories. Used in tag dispatching.
struct constant_tag {};
struct collection_tag {};
struct array_1d_tag: collection_tag {};
struct array_2d_tag: collection_tag {};
struct array_3d_tag: collection_tag {};


template<
    class T>
struct ArgumentTraits
{
};


#define CONSTANT_ARGUMENT_TRAITS(           \
    type)                                   \
template<>                                  \
struct ArgumentTraits<type>                 \
{                                           \
    typedef constant_tag argument_category; \
                                            \
    typedef type value_type;                \
};

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
