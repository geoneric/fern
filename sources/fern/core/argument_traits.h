#pragma once


namespace fern {

// Argument categories. Used in tag dispatching.
struct constant_tag {};
struct collection_tag {};


template<
    class T>
struct ArgumentTraits
{
    typedef constant_tag argument_category;

    typedef T value_type;
};


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
};

} // namespace fern
