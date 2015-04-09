// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/core/data_traits.h"


namespace fern {

template<
    class T>
typename DataTraits<T>::const_reference
                   get                 (T const& array,
                                        size_t index);

template<
    class T>
typename DataTraits<T>::reference
                   get                 (T const& array,
                                        size_t index);

} // namespace fern
