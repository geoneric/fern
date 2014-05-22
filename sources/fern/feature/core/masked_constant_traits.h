#pragma once
#include "fern/feature/core/masked_constant.h"
#include "fern/core/argument_traits.h"


namespace fern {

template<
    class T>
struct ArgumentTraits<MaskedConstant<T>>
{

    using argument_category = constant_tag;

    template<
        class U>
    struct Constant
    {
        using type = MaskedConstant<U>;
    };

    using value_type = T;

    using reference = T&;

    using const_reference = T const&;

    static bool const is_masking = true;

};


template<
    class T>
inline typename ArgumentTraits<MaskedConstant<T>>::const_reference get(
    MaskedConstant<T> const& constant)
{
    return constant.value();
}


template<
    class T>
inline typename ArgumentTraits<MaskedConstant<T>>::reference get(
    MaskedConstant<T>& constant)
{
    return constant.value();
}

} // namespace fern
