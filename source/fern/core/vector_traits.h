#pragma once
#include <cstddef>
#include <vector>
#include "fern/core/data_traits.h"


namespace fern {

template<
    class T>
struct DataTraits<
    std::vector<T>>
{

    using argument_category = array_1d_tag;

    template<
        class U>
    struct Collection
    {
        using type = std::vector<U>;
    };

    template<
        class U>
    struct Clone
    {
        using type = std::vector<U>;
    };

    // Don't use vector's typedefs. Doing it like this will make it impossible
    // to use vector<bool>, which is Good. vector<bool> is nasty since it
    // doesn't store bools. Using it works out bad in combination with threads.

    // typename std::vector<T>::value_type;
    using value_type = T;

    // typename std::vector<T>::reference;
    using reference = T&;

    // typename std::vector<T>::const_reference;
    using const_reference = T const&;

    static bool const is_masking = false;

    static size_t const rank = 1u;

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
inline typename DataTraits<std::vector<T>>::const_reference get(
    std::vector<T> const& vector,
    size_t index)
{
    return *(vector.data() + index);
}


template<
    class T>
inline typename DataTraits<std::vector<T>>::reference get(
    std::vector<T>& vector,
    size_t index)
{
    return *(vector.data() + index);
}


template<
    class U,
    class V>
inline std::vector<U> clone(
    std::vector<V> const& vector)
{
    return std::move(std::vector<U>(vector.size()));
}


template<
    class U,
    class V>
inline std::vector<U> clone(
    std::vector<V> const& vector,
    U const& value)
{
    return std::move(std::vector<U>(vector.size(), value));
}

} // namespace fern
