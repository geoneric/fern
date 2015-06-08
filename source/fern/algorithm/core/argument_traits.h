// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <type_traits>
#include <boost/mpl/if.hpp>
#include "fern/algorithm/policy/dont_mark_no_data.h"
#include "fern/algorithm/policy/skip_no_data.h"


namespace fern {
namespace algorithm {

/*!
    @ingroup    fern_algorithm_argument_traits_group
    @brief      Traits of an algorithm argument's type.
    @sa         fern_algorithm_policy_group

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
    @ingroup    fern_algorithm_argument_traits_group
    @brief      Type of an input no-data policy, given an algorithm
                argument's type.
*/
template<
    typename T>
using InputNoDataPolicyT = typename ArgumentTraits<T>::InputNoDataPolicy;


/*!
    @ingroup    fern_algorithm_argument_traits_group
    @brief      Type of an output no-data policy, given an algorithm
                argument's type.
*/
template<
    typename T>
using OutputNoDataPolicyT = typename ArgumentTraits<T>::OutputNoDataPolicy;


/*!
    @ingroup    fern_algorithm_argument_traits_group
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


/*!
    @ingroup    fern_algorithm_argument_traits_group
    @brief      Determine output no-data policy type for a temporary.
    @tparam     OutputNoDataPolicy Output no-data policy passed into the
                algorithm.
    @tparam     Argument Argument to mark no-data in.

    This alias template can be used when an algorithm is implemented in
    terms of other algorithms and these algorithms write results in
    temporary values. The output no-data policy passed into the algorithm
    itself marks no-data in the final result, but sometimes we want to
    mark no-data in temporary values. The temporary values are often
    inputs in subsequent algorithms which want to know about previously
    generated no-data.

    In case @a OutputNoDataPolicy is DontMarkNoData, then the resulting
    type is also DontMarkNoData.
*/
template<
    typename OutputNoDataPolicy,
    typename Argument>
using OutputNoDataPolicyTemporary = typename boost::mpl::if_<
    std::is_same<OutputNoDataPolicy, DontMarkNoData>,
    DontMarkNoData,
    OutputNoDataPolicyT<Argument>>::type;


/*!
    @ingroup    fern_algorithm_argument_traits_group
    @brief      Determine input no-data policy type for a temporary.
    @tparam     InputNoDataPolicy Input no-data policy passed into the
                algorithm.
    @tparam     Argument Argument to detect no-data in.

    This alias template can be used when an algorithm is implemented in
    terms of other algorithms and these algorithms write results in
    temporary values. The input no-data policy passed into the algorithm
    itself detects no-data in the original argument(s), but sometimes we
    want to detect no-data in temporary values. The temporary values are
    often inputs in subsequent algorithms which want to know about
    previously generated no-data.

    In case @a InputNoDataPolicy is SkipNoData, then the resulting
    type is also SkipNoData.
*/
template<
    typename InputNoDataPolicy,
    typename Argument>
using InputNoDataPolicyTemporary = typename boost::mpl::if_<
    std::is_same<InputNoDataPolicy, SkipNoData>,
    SkipNoData,
    InputNoDataPolicyT<Argument>>::type;

} // namespace algorithm
} // namespace fern
