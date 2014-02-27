#pragma once
#include "fern/core/argument_traits.h"
#include "fern/core/assert.h"
#include "fern/core/base_class.h"
#include "fern/algorithm/algebra/result_value.h"


namespace fern {
namespace detail {
namespace dispatch {

template<
    class A1,
    class A2,
    class RValue,
    class A1ArgumentCategory,
    class A2ArgumentCategory>
struct Result
{
};


template<
    class A1,
    class A2,
    class RValue>
struct Result<
    A1,
    A2,
    RValue,
    constant_tag,
    constant_tag>
{

    // Both argument types are not collections. The result's type equals the
    // value type.
    typedef RValue type;

};


template<
    class A1,
    class A2,
    class RValue>
struct Result<
    A1,
    A2,
    RValue,
    collection_tag,
    collection_tag>
{

    // Use collection template class of first argument as the template class
    // of the result.
    typedef typename ArgumentTraits<A1>::template Collection<RValue>::type type;

};


template<
    class A1,
    class A2,
    class RValue>
struct Result<
    A1,
    A2,
    RValue,
    constant_tag,
    collection_tag>
{

    // Use collection template class of second argument as the template class
    // of the result.
    typedef typename ArgumentTraits<A2>::template Collection<RValue>::type type;

};


template<
    class A1,
    class A2,
    class RValue>
struct Result<
    A1,
    A2,
    RValue,
    collection_tag,
    constant_tag>
{

    // Use collection template class of first argument as the template class
    // of the result.
    typedef typename ArgumentTraits<A1>::template Collection<RValue>::type type;

};

} // namespace dispatch
} // namespace detail


//! Calculate the result type of combining values of types \a A1 and \a A2.
/*!
  \tparam    A1 Type of first value to combine.
  \tparam    A2 Type of second value to combine.
  \tparam    RValue Value type of the result.

  When both of the types are collection types, the collection type of A1
  determines the collection type of the result type.

  If one of the types is a collection type, it determines the collection type
  of the result type.

  The default value type of the result is calculated by the ResultValue
  template class.

  Check the unit tests to see all this in action.
*/
template<
    class A1,
    class A2,
    class RValue=typename fern::ResultValue<
        typename ArgumentTraits<A1>::value_type,
        typename ArgumentTraits<A2>::value_type>::type>
class Result
{

private:

    FERN_STATIC_ASSERT(std::is_arithmetic, RValue)
    typedef RValue value_type;

public:

    // Determine type of result. This type differs from value_type if one of the
    // argument types is a collection.
    typedef typename detail::dispatch::Result<A1, A2, value_type,
        typename base_class<typename ArgumentTraits<A1>::argument_category,
            collection_tag>::type,
        typename base_class<typename ArgumentTraits<A2>::argument_category,
            collection_tag>::type>::type type;

};

} // namespace fern
