#pragma once
#include <cstddef>
#include <type_traits>
#include "fern/core/argument_categories.h"


namespace fern {

template<
    class T>
struct DataTraits
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
using value_type = typename DataTraits<T>::value_type;


template<
    class U,
    class V>
using Collection = typename DataTraits<U>::template Collection<V>::type;


template<
    class U,
    class V>
using clone_type = typename DataTraits<U>::template Clone<V>::type;


template<
    class T>
using const_reference = typename DataTraits<T>::const_reference;


template<
    class T>
using reference = typename DataTraits<T>::reference;


template<
    class T>
using argument_category = typename DataTraits<T>::argument_category;


template<
    class T>
struct is_masking:
    public std::integral_constant<bool, fern::DataTraits<T>::is_masking>
{};


template<
    class T>
inline constexpr size_t rank()
{
    return DataTraits<T>::rank;
}


template<
    class T>
size_t             size                (T const& value);

} // namespace fern
