// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/feature/core/data_type_traits/array.h"
#include "fern/feature/core/array_view.h"


namespace fern {

template<
    typename T,
    size_t nr_dimensions>
struct DataTypeTraits<
    ArrayView<T, nr_dimensions>>
{

    using argument_category = typename detail::dispatch::ArrayCategoryTag<T,
        nr_dimensions>::type;

    template<
        typename U>
    struct Clone
    {
        using type = ArrayView<U, nr_dimensions>;
    };

    using value_type = T;

    static bool const is_masking = false;

};

} // namespace fern
