#pragma once
#include "fern/core/argument_traits.h"


namespace fern {
namespace detail {
namespace dispatch {

template<
    class T,
    size_t nr_dimensions>
struct RasterCategoryTag
{
};


#define RASTER_CATEGORY_TAG(                    \
    nr_dimensions)                              \
template<                                       \
    class T>                                    \
struct RasterCategoryTag<T, nr_dimensions>      \
{                                               \
                                                \
    using type = raster_##nr_dimensions##d_tag; \
                                                \
};

RASTER_CATEGORY_TAG(1)
RASTER_CATEGORY_TAG(2)
RASTER_CATEGORY_TAG(3)

#undef RASTER_CATEGORY_TAG

} // namespace dispatch
} // namespace detail
} // namespace fern
