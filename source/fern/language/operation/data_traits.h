// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <stdint.h>


namespace fern {
namespace language {

struct ConstantTag { };
struct RangeTag { };
struct RasterTag: RangeTag { };


template<
    class Data>
struct DataTraits
{
    // If not specialized below, assume that the category is RangeTag.
    // Compiler error otherwise.
    using DataCategory = RangeTag;
};


#define SPECIALIZE_DATA_TRAITS_FOR_SCALAR(type) \
template<> \
struct DataTraits<type> \
{ \
    using DataCategory = ConstantTag; \
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

} // namespace language
} // namespace fern
