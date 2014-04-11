#pragma once
#include <cassert>
#include "fern/core/argument_traits.h"
#include "fern/algorithm/convolution/neighborhood/square.h"


namespace fern {

template<
    class T,
    size_t radius>
struct ArgumentTraits<
    Square<T, radius>>
{

    using argument_category = array_2d_tag;

    using value_type = T;

};


template<
    class T,
    size_t radius>
inline constexpr size_t size(
    Square<T, radius> const& /* square */)
{
    return Square<T, radius>::size();
}


template<
    class T,
    size_t radius>
inline constexpr size_t width(
    Square<T, radius> const& square)
{
    return size(square);
}


template<
    class T,
    size_t radius>
inline constexpr size_t height(
    Square<T, radius> const& square)
{
    return size(square);
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
    size_t radius>
inline T const& get(
    Square<T, radius> const& square,
    size_t index1,
    size_t index2)
{
    assert(index1 < size(square));
    assert(index2 < size(square));
    return square[index1][index2];
}

} // namespace fern
