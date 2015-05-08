// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
// For convenience, so users don't need to include it explicitly.
#include "fern/core/data_type_traits/scalar.h"
#include "fern/core/data_customization_point.h"


namespace fern {

#define CONSTANT_CUSTOMIZATION_POINT(                        \
    T)                                                       \
template<>                                                   \
inline typename DataTypeTraits<T>::const_reference get(      \
    typename DataTypeTraits<T>::value_type const& constant)  \
{                                                            \
    return constant;                                         \
}                                                            \
                                                             \
                                                             \
template<>                                                   \
inline typename DataTypeTraits<T>::reference get(            \
    typename DataTypeTraits<T>::value_type& constant)        \
{                                                            \
    return constant;                                         \
}


CONSTANT_CUSTOMIZATION_POINT(bool)
CONSTANT_CUSTOMIZATION_POINT(uint8_t)
CONSTANT_CUSTOMIZATION_POINT(uint16_t)
CONSTANT_CUSTOMIZATION_POINT(uint32_t)
CONSTANT_CUSTOMIZATION_POINT(uint64_t)
CONSTANT_CUSTOMIZATION_POINT(int8_t)
CONSTANT_CUSTOMIZATION_POINT(int16_t)
CONSTANT_CUSTOMIZATION_POINT(int32_t)
CONSTANT_CUSTOMIZATION_POINT(int64_t)
CONSTANT_CUSTOMIZATION_POINT(float)
CONSTANT_CUSTOMIZATION_POINT(double)

#undef CONSTANT_CUSTOMIZATION_POINT


template<
    typename T>
inline size_t size(
    T const& /* constant */)
{
    return 1u;
}


template<
    typename U,
    typename V>
inline U clone(
    V const& /* constant */)
{
    return U{};
}


template<
    typename U,
    typename V>
// inline U clone(
inline CloneT<V, U> clone(
    V const& /* constant */,
    U const& value)
{
    return U{value};
}

} // namespace fern
