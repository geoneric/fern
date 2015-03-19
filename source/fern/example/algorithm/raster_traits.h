#pragma once
#include <cassert>
#include "fern/core/data_traits.h"
#include "fern/example/algorithm/raster.h"


namespace fern {

/*!
    @brief      Traits used by Fern.Algorithm.
*/
template<
    typename T>
struct DataTraits<example::Raster<T>>
{

    template<
        typename U>
    struct Clone
    {
        using type = example::Raster<U>;
    };

    using value_type = T;

    using const_reference = T const&;

    using reference = T&;

    using argument_category = fern::raster_2d_tag;

    using iterator = typename example::Raster<T>::iterator;

};

} // namespace fern


namespace example {

template<
    typename T>
inline size_t size(
    Raster<T> const& raster,
    size_t index)
{
    assert(index == 0 || index == 1);
    return index == 0 ? raster.nr_rows() : raster.nr_cols();
}


template<
    typename T>
inline size_t index(
    Raster<T> const& raster,
    size_t index1,
    size_t index2)
{
    return raster.index(index1, index2);
}


template<
    typename T>
inline T const& get(
    Raster<T> const& raster,
    size_t index)
{
    return raster.get(index);
}


template<
    typename T>
inline T& get(
    Raster<T>& raster,
    size_t index)
{
    return raster.get(index);
}


template<
    typename T>
inline double cell_size(
    Raster<T> const& raster,
    size_t /* index */)
{
    return raster.cell_size();
}


template<
    typename T,
    typename U>
inline Raster<T> clone(
    Raster<U> const& raster)
{
    return std::move(Raster<T>(raster.cell_size(), raster.nr_rows(),
        raster.nr_cols()));
}


template<
    typename T>
inline typename fern::DataTraits<Raster<T>>::iterator begin(
    Raster<T>& raster)
{
    return raster.begin();
}


template<
    typename T>
inline typename fern::DataTraits<Raster<T>>::iterator end(
    Raster<T>& raster)
{
    return raster.end();
}

} // namespace example
