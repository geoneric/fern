// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/feature/core/data_type_traits/masked_scalar.h"
#include "fern/core/data_customization_point.h"


namespace fern {

template<
    typename T>
inline typename DataTypeTraits<MaskedScalar<T>>::const_reference get(
    MaskedScalar<T> const& constant)
{
    return constant.value();
}


template<
    typename T>
inline typename DataTypeTraits<MaskedScalar<T>>::reference get(
    MaskedScalar<T>& constant)
{
    return constant.value();
}

} // namespace fern
