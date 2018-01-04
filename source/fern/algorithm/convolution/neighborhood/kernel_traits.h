// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <cassert>
#include "fern/core/data_type_traits.h"
#include "fern/algorithm/convolution/neighborhood/kernel.h"


namespace fern {

template<
    class T>
struct DataTypeTraits<
    Kernel<T>>
{

    using argument_category = array_2d_tag;

    using value_type = T;

    using reference = T&;

    using const_reference = T const&;

};


template<
    class T>
inline std::size_t size(
    Kernel<T> const& kernel)
{
    return width(kernel) * height(kernel);
}


template<
    class T>
inline std::size_t size(
    Kernel<T> const& kernel,
    std::size_t /* dimension */)
{
    return kernel.size();
}


template<
    class T>
inline std::size_t width(
    Kernel<T> const& kernel)
{
    return kernel.size();
}


template<
    class T>
inline std::size_t height(
    Kernel<T> const& kernel)
{
    return kernel.size();
}


template<
    class T>
inline std::size_t radius(
    Kernel<T> const& kernel)
{
    return kernel.radius();
}


template<
    class T>
inline std::size_t index(
    Kernel<T> const& kernel,
    std::size_t index1,
    std::size_t index2)
{
    return index1 * size(kernel, 1) + index2;
}


template<
    class T>
inline typename DataTypeTraits<Kernel<T>>::const_reference get(
    Kernel<T> const& kernel,
    std::size_t index)
{
    assert(index < size(kernel));
    return kernel.weight(index);
}


template<
    class T>
inline typename DataTypeTraits<Kernel<T>>::reference get(
    Kernel<T>& kernel,
    std::size_t index)
{
    assert(index < size(kernel));
    return kernel.weight(index);
}

} // namespace fern
