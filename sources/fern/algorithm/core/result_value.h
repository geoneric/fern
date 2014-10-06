#pragma once
#include "fern/core/assert.h"
#include "fern/core/type_traits.h"
#include "fern/algorithm/core/detail/result_value.h"


namespace fern {
namespace algorithm {

/*!
    @ingroup    fern_algorithm_core_group
    @brief      Template class for calculating a result value type based on
                two argument types.
    @tparam     A1 First argument type.
    @tparam     A2 Second argument type.

    A nested @a type member is defined with the result.

    The rules for calculating the layered type are not the same as the
    ones C++ uses itself.

    The table below lists the rules implemented. The order of the types
    doesn't matter.

    A1               | A2               | type
    -----------------|------------------|---------------------
    floating point   | floating point   | largest of A1 and A2
    unsigned integer | unsigned integer | largest of A1 and A2
    singed integer   | signed integer   | largest of A1 and A2
    unsigned integer | signed integer   | see below
    floating point   | unsigned integer | A1
    floating point   | signed integer   | A1

    In case an unsigned integer and a signed integer are combined,
    the unsigned integer is looked up in a list of unsigned integer
    types with increasing size. Its index is increased by one and used
    to lookup a type in a list of signed integer types with increasing
    size. The largest of this type and the signed integer type is used
    as the result type.

    Check the unit tests to see all this in action.
*/
template<
    class A1,
    class A2>
struct ResultValue
{

    FERN_STATIC_ASSERT(std::is_arithmetic, A1)
    FERN_STATIC_ASSERT(std::is_arithmetic, A2)

    using type = typename detail::dispatch::ResultValue<A1, A2,
        typename TypeTraits<A1>::number_category,
        typename TypeTraits<A2>::number_category>::type;

};

} // namespace algorithm
} // namespace fern
