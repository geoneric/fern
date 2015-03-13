#pragma once
#include "fern/feature/core/data_traits/array.h"
#include "fern/feature/core/array_view.h"


namespace fern {

template<
    typename T,
    size_t nr_dimensions>
struct DataTraits<
    ArrayView<T, nr_dimensions>>
{

    using argument_category = typename detail::dispatch::ArrayCategoryTag<T,
        nr_dimensions>::type;

    template<
        typename U>
    struct Collection
    {
        using type = ArrayView<U, nr_dimensions>;
    };

    using value_type = T;

    static bool const is_masking = false;

};

} // namespace fern
