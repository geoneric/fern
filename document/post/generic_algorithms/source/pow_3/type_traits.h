#pragma once


template<
    typename T>
struct TypeTraits
{
};


template<
    typename T>
using value_type = typename TypeTraits<T>::value_type;
