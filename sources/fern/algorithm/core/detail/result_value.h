#pragma once
#include <boost/mpl/max_element.hpp>
#include <boost/mpl/sizeof.hpp>
#include <boost/mpl/transform_view.hpp>
#include <boost/mpl/vector.hpp>
#include "fern/core/type_traits.h"
#include "fern/core/typelist.h"


namespace fern {
namespace detail {
namespace dispatch {

//! Template class for calculating a result value type based on two argument types.
/*!
  \tparam    A1 First argument type.
  \tparam    A2 Second argument type.
  \tparam    A1NumberCategory Number category of first argument type.
  \tparam    A2NumberCategory Number category of second argument type.

  Specializations of this template class contain a nested \a type member with
  the result.
*/
template<
    class A1,
    class A2,
    class A1NumberCategory,
    class A2NumberCategory>
struct ResultValue
{
};


template<
    class A1,
    class A2>
class ResultValue<
    A1,
    A2,
    boolean_tag,
    boolean_tag>
{

public:

    using type = bool;

};


template<
    class A1,
    class A2>
class ResultValue<
    A1,
    A2,
    unsigned_integer_tag,
    unsigned_integer_tag>
{

private:

    // Pick the largest type.
    using types = boost::mpl::vector<A1, A2>;
    using iter = typename boost::mpl::max_element<boost::mpl::transform_view<
        types, boost::mpl::sizeof_<boost::mpl::_1>>>::type;

public:

    using type = typename boost::mpl::deref<typename iter::base>::type;

};


template<
    class A1,
    class A2>
class ResultValue<
    A1,
    A2,
    signed_integer_tag,
    signed_integer_tag>
{

private:

    // Pick the largest type.
    using types = boost::mpl::vector<A1, A2>;
    using iter = typename boost::mpl::max_element<boost::mpl::transform_view<
        types, boost::mpl::sizeof_<boost::mpl::_1>>>::type;

public:

    using type = typename boost::mpl::deref<typename iter::base>::type;

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
class ResultValue<
    A1,
    A2,
    unsigned_integer_tag,
    signed_integer_tag>
{

private:

    using UnsignedTypes = core::Typelist<uint8_t, uint16_t, uint32_t, uint64_t>;
    using SignedTypes = core::Typelist<int8_t, int16_t, int32_t, int64_t>;

    // Find index of A1 in list of unsigned types. Determine type of next
    // larger type in list of signed types. -> Type1
    using Type1 = typename core::at<min(
        core::find<A1, UnsignedTypes>::value + 1,
        core::size<SignedTypes>::value - 1), SignedTypes>::type;

public:

    // Result type is largest_type(Type1, A2)
    using type = typename boost::mpl::if_c<
        (sizeof(Type1) > sizeof(A2)), Type1, A2>::type;

};


template<
    class A1,
    class A2>
struct ResultValue<
    A1,
    A2,
    signed_integer_tag,
    unsigned_integer_tag>
{
    // Switch template arguments.
    using type = typename ResultValue<A2, A1, unsigned_integer_tag,
        signed_integer_tag>::type;
};


template<
    class A1,
    class A2>
class ResultValue<
    A1,
    A2,
    floating_point_tag,
    floating_point_tag>
{

private:

    // Pick the largest type.
    using types = boost::mpl::vector<A1, A2>;
    using iter = typename boost::mpl::max_element<boost::mpl::transform_view<
        types, boost::mpl::sizeof_<boost::mpl::_1>>>::type;

public:

    using type = typename boost::mpl::deref<typename iter::base>::type;

};


template<
    class A1,
    class A2>
struct ResultValue<
    A1,
    A2,
    floating_point_tag,
    signed_integer_tag>
{

    // Pick the float type.
    using type = A1;

};


template<
    class A1,
    class A2>
struct ResultValue<
    A1,
    A2,
    signed_integer_tag,
    floating_point_tag>
{

    // Pick the float type.
    using type = A2;

};


template<
    class A1,
    class A2>
struct ResultValue<
    A1,
    A2,
    floating_point_tag,
    unsigned_integer_tag>
{

    // Pick the float type.
    using type = A1;

};


template<
    class A1,
    class A2>
struct ResultValue<
    A1,
    A2,
    unsigned_integer_tag,
    floating_point_tag>
{

    // Pick the float type.
    using type = A2;

};

} // namespace dispatch
} // namespace detail
} // namespace fern
