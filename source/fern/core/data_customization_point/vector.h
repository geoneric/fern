#pragma once
#include "fern/core/data_traits/vector.h"


namespace fern {

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
