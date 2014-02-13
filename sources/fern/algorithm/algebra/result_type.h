#pragma once
#include <boost/mpl/max_element.hpp>
#include <boost/mpl/placeholders.hpp>
#include <boost/mpl/sizeof.hpp>
#include <boost/mpl/transform_view.hpp>
#include <boost/mpl/vector.hpp>
#include "fern/core/argument_traits.h"
#include "fern/core/type_traits.h"
#include "fern/core/typelist.h"


namespace fern {
namespace detail {
namespace dispatch {

template<
    class A1,
    class A2,
    class A1NumberCategory,
    class A2NumberCategory>
struct result
{
};


template<
    class A1,
    class A2>
struct result<
    A1,
    A2,
    unsigned_integer_tag,
    unsigned_integer_tag>
{
    // Pick the largest type.
    typedef boost::mpl::vector<A1, A2> types;
    typedef typename boost::mpl::max_element<boost::mpl::transform_view<types,
        boost::mpl::sizeof_<boost::mpl::_1>>>::type iter;
    typedef typename boost::mpl::deref<typename iter::base>::type type;
};


template<
    class A1,
    class A2>
struct result<
    A1,
    A2,
    signed_integer_tag, signed_integer_tag>
{
    // Pick the largest type.
    typedef boost::mpl::vector<A1, A2> types;
    typedef typename boost::mpl::max_element<boost::mpl::transform_view<types,
        boost::mpl::sizeof_<boost::mpl::_1>>>::type iter;
    typedef typename boost::mpl::deref<typename iter::base>::type type;
};


template<
    class T1,
    class T2>
constexpr auto min(
    T1 const& value1,
    T2 const& value2) -> decltype(value1 < value2 ? value1 : value2)
{
    return value1 < value2 ? value1 : value2;
}


template<
    class A1,
    class A2>
struct result<
    A1,
    A2,
    unsigned_integer_tag,
    signed_integer_tag>
{
    typedef core::Typelist<uint8_t, uint16_t, uint32_t, uint64_t> UnsignedTypes;
    typedef core::Typelist<int8_t, int16_t, int32_t, int64_t> SignedTypes;

    // Find index of A1 in list of unsigned types. Determine type of next
    // larger type in list of signed types. -> Type1
    typedef typename core::at<min(core::find<A1, UnsignedTypes>::value + 1,
        core::size<SignedTypes>::value - 1), SignedTypes>::type Type1;

    // Result type is largest_type(Type1, A2)
    typedef typename boost::mpl::if_c<
        (sizeof(Type1) > sizeof(A2)), Type1, A2>::type type;
};


template<
    class A1,
    class A2>
struct result<
    A1,
    A2,
    signed_integer_tag,
    unsigned_integer_tag>
{
    // Switch template arguments.
    typedef typename result<A2, A1, unsigned_integer_tag,
        signed_integer_tag>::type type;
};


template<
    class A1,
    class A2>
struct result<
    A1,
    A2,
    floating_point_tag,
    floating_point_tag>
{
    // Pick the largest type.
    typedef boost::mpl::vector<A1, A2> types;
    typedef typename boost::mpl::max_element<boost::mpl::transform_view<types,
        boost::mpl::sizeof_<boost::mpl::_1>>>::type iter;
    typedef typename boost::mpl::deref<typename iter::base>::type type;
};


template<
    class A1,
    class A2>
struct result<
    A1,
    A2,
    floating_point_tag,
    signed_integer_tag>
{
    // Pick the float type.
    typedef A1 type;
};


template<
    class A1,
    class A2>
struct result<
    A1,
    A2,
    signed_integer_tag,
    floating_point_tag>
{
    // Pick the float type.
    typedef A2 type;
};


template<
    class A1,
    class A2>
struct result<
    A1,
    A2,
    floating_point_tag,
    unsigned_integer_tag>
{
    // Pick the float type.
    typedef A1 type;
};


template<
    class A1,
    class A2>
struct result<
    A1,
    A2,
    unsigned_integer_tag,
    floating_point_tag>
{
    // Pick the float type.
    typedef A2 type;
};


template<
    class A1,
    class A2>
struct result<
    A1,
    A2,
    constant_tag,
    constant_tag>
{
    typedef typename result<A1, A2,
        typename TypeTraits<A1>::number_category,
        typename TypeTraits<A2>::number_category>::type type;
};


template<
    class A1,
    class A2>
struct result<
    A1,
    A2,
    collection_tag,
    collection_tag>
{
    // Use collection template class of first argument as the template class
    // of the result.
    typedef typename ArgumentTraits<A1>::value_type _vt1;
    typedef typename ArgumentTraits<A2>::value_type _vt2;
    typedef typename ArgumentTraits<A1>::template Collection<
        typename result<_vt1, _vt2,
            typename TypeTraits<_vt1>::number_category,
            typename TypeTraits<_vt2>::number_category>::type>::type type;
};


template<
    class A1,
    class A2>
struct result<
    A1,
    A2,
    constant_tag,
    collection_tag>
{
    typedef typename ArgumentTraits<A1>::value_type _vt1;
    typedef typename ArgumentTraits<A2>::value_type _vt2;
    typedef typename ArgumentTraits<A2>::template Collection<
        typename result<_vt1, _vt2,
            typename TypeTraits<_vt1>::number_category,
            typename TypeTraits<_vt2>::number_category>::type>::type type;
};


template<
    class A1,
    class A2>
struct result<
    A1,
    A2,
    collection_tag,
    constant_tag>
{
    typedef typename ArgumentTraits<A1>::value_type _vt1;
    typedef typename ArgumentTraits<A2>::value_type _vt2;
    typedef typename ArgumentTraits<A1>::template Collection<
        typename result<_vt1, _vt2,
            typename TypeTraits<_vt1>::number_category,
            typename TypeTraits<_vt2>::number_category>::type>::type type;
};

} // namespace dispatch
} // namespace detail


//! Calculate the result type of combining values of types \a A1 and \a A2.
/*!
  \tparam    A1 Type of first value to combine.
  \tparam    A2 Type of second value to combine.

  The rules for calculating the layered type are not the same as the ones C++
  uses itself.

  When both of the types are collection types, the collection type of A1
  determines the collection type of the result type.

  If one of the types is a collection type, it determines the collection type
  of the result type.

  The table below list the rules implemented with respect to value types. The
  order of the types doesn't matter.

  A1               | A2               | type                 |
  -----------------|------------------|----------------------|
  floating point   | floating point   | largest of A1 and A2 |
  unsigned integer | unsigned integer | largest of A1 and A2 |
  singed integer   | signed integer   | largest of A1 and A2 |
  unsigned integer | signed integer   | see below            |
  floating point   | unsigned integer | A1                   |
  floating point   | signed integer   | A1                   |

  In case an unsigned integer and a signed integer are combined, the unsigned
  integer is looked up in a list of unsigned integer types with increasing
  size. Its index is increased by one and used to lookup a type in a list of
  signed integer types with increasing size. The largest of this type and the
  signed integer type is used as the result type.

  Check the unit tests to see all this in action.
*/
template<
    class A1,
    class A2>
struct result
{
    // First dispatch on argument category, and then on value type.
    // In case of a compiler error, make sure the argument_category trait is
    // available for both A1 and A2.
    typedef typename detail::dispatch::result<A1, A2,
        typename ArgumentTraits<A1>::argument_category,
        typename ArgumentTraits<A2>::argument_category>::type type;
};

} // namespace fern
