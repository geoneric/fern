// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
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
