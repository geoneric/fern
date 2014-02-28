#pragma once
#include <vector>
#include "fern/core/argument_traits.h"


namespace fern {

template<
    class T>
struct ArgumentTraits<
    std::vector<T>>
{

    typedef array_1d_tag argument_category;

    template<
        class U>
    struct Collection
    {
        typedef std::vector<U> type;
    };

    typedef T value_type;

    /// typedef typename std::vector<T>::const_iterator const_iterator;

    /// typedef typename std::vector<T>::iterator iterator;

    static bool const is_masking = false;

};


template<
    class T>
size_t size(
    std::vector<T> const& vector)
{
    return vector.size();
}


template<
    class T>
T const& get(
    std::vector<T> const& vector,
    size_t index)
{
    return vector[index];
}


template<
    class T>
T& get(
    std::vector<T>& vector,
    size_t index)
{
    return vector[index];
}

} // namespace fern
