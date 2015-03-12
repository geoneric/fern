#pragma once


namespace fern {
namespace algorithm {

/*!
    @brief      Traits of an algorithm argument's type.

    - InputNoDataPolicy: Type of input no-data policy.
    - OutputNoDataPolicy: Type of output no-data policy.
    - Mask: Type of collection used to store masked elements.

    Each of these trait types can be accessed using a convenience template
    function:
    - @ref InputNoDataPolicyT
    - @ref OutputNoDataPolicyT
    - @ref MaskT
*/
template<
    typename T>
struct ArgumentTraits
{

};


/*!
    @brief      Type of an input no-data policy, given an algorithm
                argument's type.
*/
template<
    typename T>
using InputNoDataPolicyT = typename ArgumentTraits<T>::InputNoDataPolicy;


/*!
    @brief      Type of an output no-data policy, given an algorithm
                argument's type.
*/
template<
    typename T>
using OutputNoDataPolicyT = typename ArgumentTraits<T>::OutputNoDataPolicy;


/*!
    @brief      Type of a mask, given an algorithm argument's type.

    Often no-data values are special values stored in the same collection
    as the regular values. In that case, the type returned is the same as
    the type passed in: @a T.

    In case no-data values are stored in a seperate collection, like a
    bit-field, for example, then the type returned is the type of that
    collection.
*/
template<
    typename T>
using MaskT = typename ArgumentTraits<T>::Mask;

} // namespace algorithm
} // namespace fern
