// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/feature/core/masked_constant.h"
#include "fern/core/data_traits.h"


namespace fern {

template<
    typename T>
struct DataTraits<MaskedConstant<T>>
{

    using argument_category = constant_tag;

    template<
        typename U>
    struct Clone
    {
        using type = MaskedConstant<U>;
    };

    using value_type = T;

    using reference = T&;

    using const_reference = T const&;

    static bool const is_masking = true;

    static size_t const rank = 0u;

};

} // namespace fern
