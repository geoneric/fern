#pragma once
#include <cstddef>
#include "fern/core/data_traits.h"
#include "fern/feature/core/raster.h"


namespace fern {
namespace detail {
namespace dispatch {

template<
    typename T,
    size_t nr_dimensions>
struct RasterCategoryTag
{
};


#define RASTER_CATEGORY_TAG(                    \
    nr_dimensions)                              \
template<                                       \
    typename T>                                 \
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


template<
    typename T,
    size_t nr_dimensions>
struct DataTraits<
    Raster<T, nr_dimensions>>
{

    using argument_category = typename detail::dispatch::RasterCategoryTag<T,
        nr_dimensions>::type;

    template<
        typename U>
    struct Clone
    {
        using type = Raster<U, nr_dimensions>;
    };

    using value_type = T;

    using reference = T&;

    using const_reference = T const&;

    static bool const is_masking = true;

    static size_t const rank = nr_dimensions;

};

} // namespace fern
