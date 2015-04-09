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
#include "fern/algorithm/policy/input_no_data_policies.h"


namespace fern {
namespace algorithm {

/*!
    @ingroup    fern_algorithm_policy_group
    @brief      Input no-data policy that does not test for no-data.

    Use this policy whenever all input contains no-data.
*/
class AllNoData
{

public:

    static constexpr bool is_no_data   ();

    static constexpr bool is_no_data   (size_t index);

    static constexpr bool is_no_data   (size_t index1,
                                        size_t index2);

    static constexpr bool is_no_data   (size_t index1,
                                        size_t index2,
                                        size_t index3);

                   AllNoData           (AllNoData const&)=delete;

                   AllNoData           (AllNoData&&)=default;

    AllNoData&
                   operator=           (AllNoData const&)=delete;

    AllNoData&     operator=           (AllNoData&&)=default;

                   ~AllNoData          ()=default;

};


/*!
    @brief      Return whether input is no-data.

    This method is called in case of a 0D input.
*/
inline constexpr bool AllNoData::is_no_data()
{
    return true;
}


/*!
    @brief      Return whether input is no-data.
    @param      index Index of element to test.

    This method is called in case of a 1D input.
*/
inline constexpr bool AllNoData::is_no_data(
    size_t /* index */)
{
    return true;
}


/*!
    @brief      Return whether input is no-data.
    @param      index1 Index of first dimension of element to test.
    @param      index2 Index of second dimension of element to test.

    This method is called in case of a 2D input.
*/
inline constexpr bool AllNoData::is_no_data(
    size_t /* index1 */,
    size_t /* index2 */)
{
    return true;
}


/*!
    @brief      Return whether input is no-data.
    @param      index1 Index of first dimension of element to test.
    @param      index2 Index of second dimension of element to test.
    @param      index3 Index of third dimension of element to test.

    This method is called in case of a 3D input.
*/
inline constexpr bool AllNoData::is_no_data(
    size_t /* index1 */,
    size_t /* index2 */,
    size_t /* index3 */)
{
    return true;
}

} // namespace algorithm
} // namespace fern
