#pragma once
#include <cstddef>
#include "fern/algorithm/policy/input_no_data_policies.h"


namespace fern {
namespace algorithm {

/*!
    @ingroup    fern_algorithm_policy_group
    @brief      Input no-data policy that does not test for no-data.
    @tparam     ArgumentNoDataPolicies Input no-data policies of the
                algorithm's arguments.

    Use this policy whenever the input does not contain no-data.
*/
template<
    class... ArgumentNoDataPolicies>
class SkipNoData:
    public InputNoDataPolicies<ArgumentNoDataPolicies...>
{

public:

    static constexpr bool is_no_data   ();

    static constexpr bool is_no_data   (size_t index);

    static constexpr bool is_no_data   (size_t index1,
                                        size_t index2);

    static constexpr bool is_no_data   (size_t index1,
                                        size_t index2,
                                        size_t index3);

                   SkipNoData          (ArgumentNoDataPolicies&&... policies);

                   SkipNoData          (SkipNoData const&)=delete;

                   SkipNoData          (SkipNoData&&)=default;

    SkipNoData&
                   operator=           (SkipNoData const&)=delete;

    SkipNoData&    operator=           (SkipNoData&&)=default;

                   ~SkipNoData         ()=default;

};


/*!
    @brief      Return whether input is no-data.

    This method is called in case of a 0D input.
*/
template<
    class... ArgumentNoDataPolicies>
inline constexpr bool SkipNoData<ArgumentNoDataPolicies...>::is_no_data()
{
    return false;
}


/*!
    @brief      Return whether input is no-data.
    @param      index Index of element to test.

    This method is called in case of a 1D input.
*/
template<
    class... ArgumentNoDataPolicies>
inline constexpr bool SkipNoData<ArgumentNoDataPolicies...>::is_no_data(
    size_t /* index */)
{
    return false;
}


/*!
    @brief      Return whether input is no-data.
    @param      index1 Index of first dimension of element to test.
    @param      index2 Index of second dimension of element to test.

    This method is called in case of a 2D input.
*/
template<
    class... ArgumentNoDataPolicies>
inline constexpr bool SkipNoData<ArgumentNoDataPolicies...>::is_no_data(
    size_t /* index1 */,
    size_t /* index2 */)
{
    return false;
}


/*!
    @brief      Return whether input is no-data.
    @param      index1 Index of first dimension of element to test.
    @param      index2 Index of second dimension of element to test.
    @param      index3 Index of third dimension of element to test.

    This method is called in case of a 3D input.
*/
template<
    class... ArgumentNoDataPolicies>
inline constexpr bool SkipNoData<ArgumentNoDataPolicies...>::is_no_data(
    size_t /* index1 */,
    size_t /* index2 */,
    size_t /* index3 */)
{
    return false;
}


/*!
    @brief      Constructor.
    @param      policies Policies of inputs.
    @sa         fern::algorithm::core::unite_no_data,
                fern::algorithm::core::intersect_no_data

    Most algorithms don't need input no-data policies of each individual
    input. Often, they only need to know the union of the input no-data.
*/
template<
    class... ArgumentNoDataPolicies>
inline SkipNoData<ArgumentNoDataPolicies...>::SkipNoData(
    ArgumentNoDataPolicies&&... policies)

    : InputNoDataPolicies<ArgumentNoDataPolicies...>(
          std::forward<ArgumentNoDataPolicies>(policies)...)

{
}

} // namespace algorithm
} // namespace fern
