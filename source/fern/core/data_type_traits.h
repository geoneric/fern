// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <cstddef>
#include <type_traits>
#include "fern/core/argument_categories.h"


namespace fern {

/*!
    @ingroup    fern_data_type_traits_group
    @brief      Traits of a data type.
*/
template<
    class T>
struct DataTypeTraits
{

    //! By default, we grab T's value type. Specialize if needed.
    using value_type = typename T::value_type;

    //! By default, we grab T's reference type. Specialize if needed.
    using reference = typename T::reference;

    //! By default, we grab T's const_reference type. Specialize if needed.
    using const_reference = typename T::const_reference;

};


/*!
    @ingroup    fern_data_type_traits_group
    @brief      Type of an individual value / element in the data type.

    For simple data types, like integers, the value type equals the data type.
    In other cases the data type instance contains one or more value type
    instances. For example, a masked constant can contain an integer and a
    vector can contain a number of integers.
*/
template<
    class T>
using value_type = typename DataTypeTraits<T>::value_type;


/*!
    @ingroup    fern_data_type_traits_group
    @brief      The same data type as @a DataType, but with the value type
                @a ValueType.

    @code
    // Foo is of type std::vector<double>.
    CloneT<std::vector<int>, double> foo;
    @endcode
*/
template<
    class DataType,
    class ValueType>
using CloneT = typename DataTypeTraits<DataType>::template Clone<ValueType>::type;


/*!
    @ingroup    fern_data_type_traits_group
*/
template<
    class T>
using const_reference = typename DataTypeTraits<T>::const_reference;


/*!
    @ingroup    fern_data_type_traits_group
*/
template<
    class T>
using reference = typename DataTypeTraits<T>::reference;


/*!
    @ingroup    fern_data_type_traits_group
*/
template<
    class T>
using argument_category = typename DataTypeTraits<T>::argument_category;


/*!
    @ingroup    fern_data_type_traits_group
*/
template<
    class T>
struct is_masking:
    public std::integral_constant<bool, fern::DataTypeTraits<T>::is_masking>
{};


/*!
    @ingroup    fern_data_type_traits_group
    @brief      Return the dimensionality of @a T.

    A 2D array has rank 2, a std::vector has rank 1, a constant has rank 0.
*/
template<
    class T>
inline constexpr size_t rank()
{
    return DataTypeTraits<T>::rank;
}

} // namespace fern
