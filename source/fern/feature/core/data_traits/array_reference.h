#pragma once
#include "fern/core/data_traits.h"
#include "fern/feature/core/array_reference.h"


namespace fern {
namespace detail {
namespace dispatch {

template<
    class T,
    size_t nr_dimensions>
struct ArrayReferenceCategoryTag
{
};


#define ARRAY_CATEGORY_TAG(                         \
    nr_dimensions)                                  \
template<                                           \
    class T>                                        \
struct ArrayReferenceCategoryTag<T, nr_dimensions>  \
{                                                   \
                                                    \
    using type = array_##nr_dimensions##d_tag;      \
                                                    \
};

ARRAY_CATEGORY_TAG(1)
ARRAY_CATEGORY_TAG(2)
ARRAY_CATEGORY_TAG(3)

#undef ARRAY_CATEGORY_TAG

} // namespace dispatch
} // namespace detail


// template<
//     class T,
//     size_t nr_dimensions>
// struct DataTraits<
//     View<T, nr_dimensions>>
// {
// 
//     using argument_category = typename detail::dispatch::ArrayReferenceCategoryTag<T, nr_dimensions>::type;
// 
//     template<
//         class U>
//     struct Collection
//     {
//         using type = ArrayReference<T, nr_dimensions>;
//     };
// 
//     using value_type = T;
// 
// };


template<
    class T,
    size_t nr_dimensions>
struct DataTraits<
    ArrayReference<T, nr_dimensions>>
{

    using argument_category = typename
        detail::dispatch::ArrayReferenceCategoryTag<T, nr_dimensions>::type;

    /// template<
    ///     class U>
    /// struct Collection
    /// {
    ///     using type = ArrayReference<U, nr_dimensions>;
    /// };

    /// template<
    ///     class U>
    /// struct Clone
    /// {
    ///     using type = Array<U, nr_dimensions>;
    /// };

    using value_type = T;

    using reference = T&;

    using const_reference = T const&;

    static bool const is_masking = false;

    static size_t const rank = nr_dimensions;

};

} // namespace fern
