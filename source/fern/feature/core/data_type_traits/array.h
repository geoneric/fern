// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/core/data_type_traits.h"
#include "fern/feature/core/array.h"


namespace fern {
namespace detail {
namespace dispatch {

template<
    typename T,
    size_t nr_dimensions>
struct ArrayCategoryTag
{
};


#define ARRAY_CATEGORY_TAG(                     \
    nr_dimensions)                              \
template<                                       \
    typename T>                                 \
struct ArrayCategoryTag<T, nr_dimensions>       \
{                                               \
                                                \
    using type = array_##nr_dimensions##d_tag;  \
                                                \
};

ARRAY_CATEGORY_TAG(1)
ARRAY_CATEGORY_TAG(2)
ARRAY_CATEGORY_TAG(3)

#undef ARRAY_CATEGORY_TAG

} // namespace dispatch
} // namespace detail


template<
    typename T,
    size_t nr_dimensions>
struct DataTypeTraits<
    Array<T, nr_dimensions>>
{

    using argument_category = typename detail::dispatch::ArrayCategoryTag<T,
        nr_dimensions>::type;

    template<
        typename U>
    struct Clone
    {
        using type = Array<U, nr_dimensions>;
    };

    using value_type = T;

    using reference = T&;

    using const_reference = T const&;

    static bool const is_masking = false;

    static size_t const rank = nr_dimensions;

};

} // namespace fern
