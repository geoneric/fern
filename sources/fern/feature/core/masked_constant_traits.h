#pragma once
#include "fern/feature/core/masked_constant.h"
#include "fern/core/argument_traits.h"


namespace fern {

template<
    class T>
struct ArgumentTraits<MaskedConstant<T>>
{

    typedef constant_tag argument_category;

    template<
        class U>
    struct Constant
    {
        typedef MaskedConstant<U> type;
    };

    typedef T value_type;

    static bool const is_masking = true;

};


template<
    class T>
inline T const& get(
    MaskedConstant<T> const& constant)
{
    return constant.value();
}


template<
    class T>
inline T& get(
    MaskedConstant<T>& constant)
{
    return constant.value();
}

} // namespace fern
