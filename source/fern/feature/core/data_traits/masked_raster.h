#pragma once
#include "fern/feature/core/data_traits/raster.h"
#include "fern/feature/core/masked_raster.h"


namespace fern {

template<
    typename T,
    size_t nr_dimensions>
struct DataTraits<
    MaskedRaster<T, nr_dimensions>>
{

    using argument_category = typename detail::dispatch::RasterCategoryTag<T,
        nr_dimensions>::type;

    template<
        typename U>
    struct Clone
    {
        using type = MaskedRaster<U, nr_dimensions>;
    };

    using value_type = T;

    using reference = T&;

    using const_reference = T const&;

    static bool const is_masking = true;

    static size_t const rank = nr_dimensions;

};

} // namespace fern
