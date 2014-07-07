#pragma once

// Inspiration: http://stackoverflow.com/questions/14261183/how-to-make-generic-computations-over-heterogeneous-argument-packs-of-a-variadic


namespace fern {

template<
    int index,
    typename... Types
>
struct nth_type
{
};


template<
    typename T,
    typename... Types
>
struct nth_type<0, T, Types...>
{
    using type = T;
};


template<
    int index,
    typename T,
    typename... Types
>
struct nth_type<index, T, Types...>
{
    using type = typename nth_type<index - 1, Types...>::type;
};


template<
    typename... Types>
struct first_type
{
    using type = typename nth_type<0, Types...>::type;
};


template<
    typename... Types
>
struct last_type
{
    using type = typename nth_type<sizeof...(Types) - 1, Types...>::type;
};

} // namespace fern
