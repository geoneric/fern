#pragma once


namespace fern {

// Argument categories. Used in tag dispatching.
struct constant_tag {};
struct collection_tag {};


template<
    class T>
struct ArgumentTraits
{
};


#define CONSTANT_ARGUMENT_TRAITS(           \
    type)                                   \
template<>                                  \
struct ArgumentTraits<type>                 \
{                                           \
    typedef constant_tag argument_category; \
                                            \
    typedef type value_type;                \
};

CONSTANT_ARGUMENT_TRAITS(uint8_t)
CONSTANT_ARGUMENT_TRAITS(uint16_t)
CONSTANT_ARGUMENT_TRAITS(uint32_t)
CONSTANT_ARGUMENT_TRAITS(uint64_t)
CONSTANT_ARGUMENT_TRAITS(int8_t)
CONSTANT_ARGUMENT_TRAITS(int16_t)
CONSTANT_ARGUMENT_TRAITS(int32_t)
CONSTANT_ARGUMENT_TRAITS(int64_t)
CONSTANT_ARGUMENT_TRAITS(float)
CONSTANT_ARGUMENT_TRAITS(double)

#undef CONSTANT_ARGUMENT_TRAITS


template<
    class T>
struct ArgumentTraits<
    std::vector<T>>
{
    typedef collection_tag argument_category;

    template<
        class U>
    struct Collection
    {
        typedef std::vector<U> type;
    };

    typedef T value_type;

    typedef typename std::vector<T>::const_iterator const_iterator;

    typedef typename std::vector<T>::iterator iterator;

};


template<
    class U,
    class V>
void resize(
    std::vector<U>& vector,
    std::vector<V> const& other_vector)
{
    vector.resize(other_vector.size());
}


template<
    class T>
typename ArgumentTraits<std::vector<T>>::const_iterator begin(
    std::vector<T> const& vector)
{
    return vector.begin();
}


template<
    class T>
typename ArgumentTraits<std::vector<T>>::iterator begin(
    std::vector<T>& vector)
{
    return vector.begin();
}


template<
    class T>
typename ArgumentTraits<std::vector<T>>::const_iterator end(
    std::vector<T> const& vector)
{
    return vector.end();
}


template<
    class T>
typename ArgumentTraits<std::vector<T>>::iterator end(
    std::vector<T>& vector)
{
    return vector.end();
}

} // namespace fern
