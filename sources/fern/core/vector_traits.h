#pragma once
#include <cstddef>
#include <vector>
#include "fern/core/argument_traits.h"


namespace fern {

template<
    class T>
struct ArgumentTraits<
    std::vector<T>>
{

    using argument_category = array_1d_tag;

    template<
        class U>
    struct Collection
    {
        using type = std::vector<U>;
    };

    using value_type = typename std::vector<T>::value_type;

    using reference = typename std::vector<T>::reference;

    using const_reference = typename std::vector<T>::const_reference;

    static bool const is_masking = false;

};


template<
    class T>
inline size_t size(
    std::vector<T> const& vector)
{
    return vector.size();
}


template<
    class T>
inline typename ArgumentTraits<std::vector<T>>::const_reference get(
    std::vector<T> const& vector,
    size_t index)
{
    return vector[index];
}


template<
    class T>
inline typename ArgumentTraits<std::vector<T>>::reference get(
    std::vector<T>& vector,
    size_t index)
{
    return vector[index];
}

} // namespace fern
