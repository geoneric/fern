#pragma once
#include "fern/feature/core/data_traits/masked_constant.h"
#include "fern/core/data_customization_point.h"


namespace fern {

template<
    typename T>
inline typename DataTraits<MaskedConstant<T>>::const_reference get(
    MaskedConstant<T> const& constant)
{
    return constant.value();
}


template<
    typename T>
inline typename DataTraits<MaskedConstant<T>>::reference get(
    MaskedConstant<T>& constant)
{
    return constant.value();
}

} // namespace fern
