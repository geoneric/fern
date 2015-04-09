// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <tuple>
#include <boost/mpl/if.hpp>


namespace fern {
namespace core {
namespace mpl = boost::mpl;

template<
    class... TYPES>
struct Typelist
{
};


template<
    class... TYPES>
struct size
{
};


template<
    class... TYPES>
struct size<
    Typelist<TYPES...>>
{
    enum {
        value = sizeof...(TYPES)
    };
};


template<
    class T,
    class... TYPES>
struct push_front
{
};


template<
    class T,
    class... TYPES>
struct push_front<
    T,
    Typelist<TYPES...>>
{
    using type = Typelist<T, TYPES... >;
};


template<
    class... TYPES>
struct pop_front
{
};


template<
    class T,
    class... TYPES>
struct pop_front<
    Typelist<T, TYPES...>>
{
    using type = Typelist<TYPES...>;
};


template<
    size_t N,
    class... TYPES>
struct at
{
};


template<
    size_t N,
    class... TYPES>
struct at<
    N,
    Typelist<TYPES...>>
{
    using type = typename std::tuple_element<N, std::tuple<TYPES...>>::type;
};


namespace detail {

template<
    class TypeToFind,
    size_t index,
    class... TYPES>
struct find_from
{
};


template<
    class TypeToFind,
    size_t index,
    class... TYPES>
struct find_from<
    TypeToFind,
    index,
    Typelist<TYPES...>>
{
    using type = typename mpl::if_c<
        // If the current type equals the type to look for ...
        std::is_same<typename at<index, Typelist<TYPES...>>::type,
            TypeToFind>::value,
        // ... use a type to represent the current index, ...
        mpl::int_<index>,
        // ... else try the next type.
        find_from<TypeToFind, index + 1, Typelist<TYPES...>>
    >::type;

    enum {
        value = type::value
    };
};

} // namespace detail


template<
    class TypeToFind,
    class... TYPES>
struct find
{
};


template<
    class TypeToFind,
    class... TYPES>
struct find<
    TypeToFind,
    Typelist<TYPES...>>
{
    enum {
        value = detail::find_from<TypeToFind, 0, Typelist<TYPES...>>::value
    };
};

} // namespace core
} // namespace fern
