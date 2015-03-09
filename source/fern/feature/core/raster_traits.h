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
    struct Collection
    {
        using type = Raster<U, nr_dimensions>;
    };

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


template<
    typename T,
    size_t nr_dimensions>
inline size_t size(
    Raster<T, nr_dimensions> const& raster)
{
    return raster.num_elements();
}


template<
    typename T,
    size_t nr_dimensions>
inline size_t size(
    Raster<T, nr_dimensions> const& raster,
    size_t dimension)
{
    assert(dimension < raster.num_dimensions());
    return raster.shape()[dimension];
}


template<
    typename T,
    size_t nr_dimensions>
inline size_t index(
    Raster<T, nr_dimensions> const& raster,
    size_t index1,
    size_t index2)
{
    return index1 * size(raster, 1) + index2;
}


template<
    typename T,
    size_t nr_dimensions>
inline typename DataTraits<Raster<T, nr_dimensions>>::const_reference get(
    Raster<T, nr_dimensions> const& raster,
    size_t index)
{
    assert(index < raster.num_elements());
    return raster.data()[index];
}


template<
    typename T,
    size_t nr_dimensions>
inline typename DataTraits<Raster<T, nr_dimensions>>::reference get(
    Raster<T, nr_dimensions>& raster,
    size_t index)
{
    assert(index < raster.num_elements());
    return raster.data()[index];
}


template<
    typename U,
    typename V>
inline Raster<U, 1> clone(
    Raster<V, 1> const& raster)
{
    return std::move(Raster<U, 1>(extents[raster.shape()[0]],
        raster.transformation()));
}


template<
    typename U,
    typename V>
inline Raster<U, 1> clone(
    Raster<V, 1> const& raster,
    U const& value)
{
    return std::move(Raster<U, 1>(extents[raster.shape()[0]],
        raster.transformation(),
        value));
}


template<
    typename T>
inline double cell_size(
    Raster<T, 2> const& raster,
    size_t index)
{
    assert(index == 0 || index == 1);
    return raster.transformation()[index == 0 ? 1 : 3];
}


template<
    typename T>
inline double cell_area(
    Raster<T, 2> const& raster)
{
    return raster.transformation()[1] * raster.transformation()[3];
}


template<
    typename U,
    typename V>
inline Raster<U, 2> clone(
    Raster<V, 2> const& raster)
{
    return std::move(Raster<U, 2>(
        extents[raster.shape()[0]][raster.shape()[1]],
        raster.transformation()));
}


template<
    typename U,
    typename V>
inline Raster<U, 2> clone(
    Raster<V, 2> const& raster,
    U const& value)
{
    return std::move(Raster<U, 2>(
        extents[raster.shape()[0]][raster.shape()[1]],
        raster.transformation(),
        value));
}


template<
    typename U,
    typename V>
inline Raster<U, 3> clone(
    Raster<V, 3> const& raster)
{
    return std::move(Raster<U, 3>(
        extents[raster.shape()[0]][raster.shape()[1],raster.shape()[2]],
        raster.transformation()));
}


template<
    typename U,
    typename V>
inline Raster<U, 3> clone(
    Raster<V, 3> const& raster,
    U const& value)
{
    return std::move(Raster<U, 3>(
        extents[raster.shape()[0]][raster.shape()[1],raster.shape()[2]],
        raster.transformation(),
        value));
}

} // namespace fern
