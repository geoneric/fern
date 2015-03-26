#pragma once
#include "fern/core/data_traits.h"
#include "fern/python/feature/detail/masked_raster.h"


namespace fern {

template<
    typename T>
struct DataTraits<
    python::detail::MaskedRaster<T>>
{

    using argument_category = raster_2d_tag;

    template<
        typename U>
    struct Clone
    {
        using type = python::detail::MaskedRaster<U>;
    };

    using value_type = T;

    using reference = T&;

    using const_reference = T const&;

    static bool const is_masking = true;

    static size_t const rank = 2;

};

} // namespace fern
