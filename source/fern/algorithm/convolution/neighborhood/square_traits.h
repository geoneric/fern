#pragma once
#include <cassert>
#include "fern/core/data_traits.h"
#include "fern/algorithm/convolution/neighborhood/square.h"


namespace fern {

template<
    class T,
    size_t radius>
struct DataTraits<
    Square<T, radius>>
{

    using argument_category = array_2d_tag;

    using value_type = T;

    using reference = T&;

    using const_reference = T const&;

};


template<
    class T,
    size_t radius>
inline constexpr size_t size(
    Square<T, radius> const& square)
{
    return width(square) * height(square);
}


template<
    class T,
    size_t radius>
inline constexpr size_t size(
    Square<T, radius> const& /* square */,
    size_t /* dimension */)
{
    return Square<T, radius>::size();
}


template<
    class T,
    size_t radius>
inline constexpr size_t width(
    Square<T, radius> const& /* square */)
{
    return Square<T, radius>::size();
}


template<
    class T,
    size_t radius>
inline constexpr size_t height(
    Square<T, radius> const& /* square */)
{
    return Square<T, radius>::size();
}


template<
    class T,
    size_t radius_>
inline constexpr size_t radius(
    Square<T, radius_> const& /* square */)
{
    return radius_;
}


template<
    class T,
    size_t radius_>
inline constexpr size_t index(
    Square<T, radius_> const& square,
    size_t index1,
    size_t index2)
{
    return index1 * size(square, 1) + index2;
}


template<
    class T,
    size_t radius>
inline typename DataTraits<Square<T, radius>>::const_reference get(
    Square<T, radius> const& square,
    size_t index)
{
    assert(index < size(square));
    return square.weight(index);
}


template<
    class T,
    size_t radius>
inline typename DataTraits<Square<T, radius>>::reference get(
    Square<T, radius>& square,
    size_t index)
{
    assert(index < size(square));
    return square.weight(index);
}

} // namespace fern
