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
};


template<
    class T>
struct ArgumentTraits<
    std::vector<T>>
{
    typedef collection_tag argument_category;
};

} // namespace fern
