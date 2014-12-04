#pragma once
#include <type_traits>
#include <boost/mpl/if.hpp>


namespace fern {
namespace detail {

//! Determine which of the \a CLASSES passed in is a base class of \a Class.
/*!
  \tparam    Class Class to check.
  \tparam    CLASSES Classes to try as base class.

  A nested type called \a type is set to the result. If none of the classes in
  \a CLASSES is a base class of \a Class, then type is set to \a Class. If
  multiple classes in \a CLASSES is a base class of \a Class, then type is set
  to the first one.
*/
template<
    class Class,
    class... CLASSES>
struct base_class
{
};


template<
    class Class,
    class BaseClass>
struct base_class<
    Class,
    BaseClass>
{
    // This template matches if there is only one type left to test.
    using type = typename boost::mpl::if_<
        // If BaseClass is a base of Class ...
        typename std::is_base_of<BaseClass, Class>::type,
        // ... use BaseClass as the result type ...
        BaseClass,
        // ... else, use the Class type itself and be done with it.
        Class
    >::type;
};


template<
    class Class,
    class BaseClass,
    class... CLASSES>
struct base_class<
    Class,
    BaseClass,
    CLASSES...>
{
    using type = typename boost::mpl::if_<
        // If BaseClass is a base of Class ...
        typename std::is_base_of<BaseClass, Class>::type,
        // ... use BaseClass as the result type ...
        BaseClass,
        // ... else, try the next tag type.
        typename base_class<Class, CLASSES...>::type
    >::type;
};

} // namespace detail


template<
    class Class,
    class... CLASSES>
using base_class = typename detail::base_class<Class, CLASSES...>::type;

} // namespace fern
