#pragma once


struct number_tag {};
struct raster_tag {};


template<
    typename T>
struct TypeTraits
{
    using tag = number_tag;  // Default.
};


template<
    typename T>
using tag = typename TypeTraits<T>::tag;


template<
    typename T>
using value_type = typename TypeTraits<T>::value_type;
