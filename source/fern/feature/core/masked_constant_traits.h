#pragma once
#include "fern/feature/core/masked_constant.h"
#include "fern/core/argument_traits.h"


namespace fern {

template<
    typename T>
struct ArgumentTraits<MaskedConstant<T>>
{

    using argument_category = constant_tag;

    template<
        typename U>
    struct Constant
    {
        using type = MaskedConstant<U>;
    };

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


template<
    typename T>
inline typename ArgumentTraits<MaskedConstant<T>>::const_reference get(
    MaskedConstant<T> const& constant)
{
    return constant.value();
}


template<
    typename T>
inline typename ArgumentTraits<MaskedConstant<T>>::reference get(
    MaskedConstant<T>& constant)
{
    return constant.value();
}

} // namespace fern
