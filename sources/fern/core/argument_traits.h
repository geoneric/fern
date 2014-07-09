#pragma once
#include "fern/core/argument_categories.h"


namespace fern {

template<
    class T>
struct ArgumentTraits
{

    //! By default, we grab T's value type. Specialize if needed.
    using value_type = typename T::value_type;

    //! By default, we grab T's reference type. Specialize if needed.
    using reference = typename T::reference;

    //! By default, we grab T's const_reference type. Specialize if needed.
    using const_reference = typename T::const_reference;

};


template<
    class T>
using value_type = typename ArgumentTraits<T>::value_type;


template<
    class U,
    class V>
using Collection = typename ArgumentTraits<U>::template Collection<V>::type;


template<
    class T>
using const_reference = typename ArgumentTraits<T>::const_reference;


template<
    class T>
using reference = typename ArgumentTraits<T>::reference;


template<
    class T>
using argument_category = typename ArgumentTraits<T>::argument_category;

} // namespace fern
