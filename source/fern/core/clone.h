#pragma once
#include "fern/core/data_traits.h"


namespace fern {

//! Clone a collection, using a different value type.
/*!
    \tparam    ValueType Value type to use for the result.
    \tparam    Value Value whose collection type is used for the result.
    \param     value Instance whose properties are used for the result.
    \return    New collection with value type \a ValueType and the same
               collection properties as \a value.
    \warning   The individual values in \a value are not copied to the result.

    The result of calling this function is useful as an output parameter when
    calling an algorithm.

    For example, result of calling

    \code
    auto result = clone<double>(std::vector<int>{1, 2, 3, 4, 5});
    \endcode

    is a `std::vector<double>` with the same lenght. All values are initialized
    by the `std::vector` constructor used in the implementation of clone for
    `std::vector`.
*/
template<
    class ValueType,
    class Value>
Collection<Value, ValueType>
                   clone               (Value const& value);


/*!
    \overload
*/
template<
    class ValueType,
    class Value>
Collection<Value, ValueType>
                   clone               (Value const& value,
                                        ValueType const& initial_value);

} // namespace fern
