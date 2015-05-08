// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <boost/mpl/and.hpp>
#include <boost/mpl/if.hpp>
#include "fern/core/data_type_traits/scalar.h"
#include "fern/core/assert.h"
#include "fern/core/base_class.h"
#include "fern/algorithm/core/result_value.h"


namespace fern {
namespace algorithm {
namespace detail {
namespace dispatch {

template<
    typename A1,
    typename A2,
    typename RValue,
    typename A1ArgumentCategory,
    typename A2ArgumentCategory>
struct Result
{
};


template<
    typename A1,
    typename A2,
    typename RValue>
struct Result<
    A1,
    A2,
    RValue,
    scalar_tag,
    scalar_tag>
{

    /// // Both argument types are not collections. The result's type equals the
    /// // value type.
    /// using type = RValue;

    using type = typename boost::mpl::if_<
        typename boost::mpl::and_<
            typename std::is_arithmetic<A1>::type,
            typename std::is_arithmetic<A2>::type>::type,
        // Both argument types are arithmetic types. The result's type equals
        // the value type.
        RValue,
        typename boost::mpl::if_<
            typename std::is_arithmetic<A1>::type,

            // A2 is a type representing a constant. Select its template class
            // in combination with the result value type.
            CloneT<A2, RValue>,

            // A1 is a type representing a constant. Select its template class
            // in combination with the result value type.
            CloneT<A1, RValue>>::type
    >::type;

};


template<
    typename A1,
    typename A2,
    typename RValue>
struct Result<
    A1,
    A2,
    RValue,
    collection_tag,
    collection_tag>
{

    // Use collection template class of first argument as the template class
    // of the result.
    using type = CloneT<A1, RValue>;

};


template<
    typename A1,
    typename A2,
    typename RValue>
struct Result<
    A1,
    A2,
    RValue,
    scalar_tag,
    collection_tag>
{

    // Use collection template class of second argument as the template class
    // of the result.
    using type = CloneT<A2, RValue>;

};


template<
    typename A1,
    typename A2,
    typename RValue>
struct Result<
    A1,
    A2,
    RValue,
    collection_tag,
    scalar_tag>
{

    // Use collection template class of first argument as the template class
    // of the result.
    using type = CloneT<A1, RValue>;

};

} // namespace dispatch
} // namespace detail


/*!
   @ingroup     fern_algorithm_core_group
   @brief       Calculate the result type of combining values of types @a A1
                and @a A2.
   @tparam      A1 Type of first value to combine.
   @tparam      A2 Type of second value to combine.
   @tparam      RValue Value type of the result.

   When both of the types are collection types, the collection type of A1
   determines the collection type of the result type.

   If one of the types is a collection type, it determines the collection type
   of the result type.

   The default value type of the result is calculated by the ResultValue
   template class.

   Check the unit tests to see all this in action.
*/
template<
    typename A1,
    typename A2,
    typename RValue=typename ResultValue<
        typename DataTypeTraits<A1>::value_type,
        typename DataTypeTraits<A2>::value_type>::type>
class Result
{

private:

    FERN_STATIC_ASSERT(std::is_arithmetic, RValue)
    using value_type = RValue;

public:

    // Determine type of result. This type differs from value_type if one of the
    // argument types is a collection.
    using type = typename detail::dispatch::Result<A1, A2, value_type,
        base_class<argument_category<A1>, collection_tag>,
        base_class<argument_category<A2>, collection_tag>>::type;

};


template<
    typename A1,
    typename A2>
using result_type = typename Result<A1, A2>::type;

} // namespace algorithm
} // namespace fern
