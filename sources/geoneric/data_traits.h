#pragma once
#include <stdint.h>


namespace geoneric {

struct ScalarTag { };
struct RangeTag { };
struct RasterTag: RangeTag { };


template<
    class Data>
struct DataTraits
{
    // If not specialized below, assume that the category is RangeTag.
    // Compiler error otherwise.
    typedef RangeTag DataCategory;
};


#define SPECIALIZE_DATA_TRAITS_FOR_SCALAR(type) \
template<> \
struct DataTraits<type> \
{ \
    typedef ScalarTag DataCategory; \
};

SPECIALIZE_DATA_TRAITS_FOR_SCALAR(int8_t)
SPECIALIZE_DATA_TRAITS_FOR_SCALAR(int16_t)
SPECIALIZE_DATA_TRAITS_FOR_SCALAR(int32_t)
SPECIALIZE_DATA_TRAITS_FOR_SCALAR(int64_t)
SPECIALIZE_DATA_TRAITS_FOR_SCALAR(uint8_t)
SPECIALIZE_DATA_TRAITS_FOR_SCALAR(uint16_t)
SPECIALIZE_DATA_TRAITS_FOR_SCALAR(uint32_t)
SPECIALIZE_DATA_TRAITS_FOR_SCALAR(uint64_t)
SPECIALIZE_DATA_TRAITS_FOR_SCALAR(float)
SPECIALIZE_DATA_TRAITS_FOR_SCALAR(double)

#undef SPECIALIZE_DATA_TRAITS_FOR_SCALAR

} // namespace geoneric
