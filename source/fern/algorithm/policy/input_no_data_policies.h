// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <tuple>
#include <utility>


namespace fern {
namespace algorithm {

/*!
    @ingroup    fern_algorithm_policy_group
    @brief      Base class for storing input no-data policies of algorithm
                arguments.
    @tparam     NoDataPolicies For each input no-data policy its class.
    @sa         @ref fern_algorithm_policies

    Most algorithms don't need to be able to test whether elements from their
    arguments have a valid value or not. In these cases the algorithm's result
    can be preprocessed to contain the union of the no-data elements.

    But some algorithms, like fern::algorithm::core::cover, do need
    to be able to test whether elements from their arguments have a
    valid value or not. In these cases the input no-data policy passed
    to the algorithms must contain input no-data policies for each
    argument. Input no-data policy classes that inherit from this class
    can store and retrieve this information. It is allowed to pass an
    empty set of classes in @a NoDataPolicies. That way, the single
    input no-data policy class can be used for both kinds of algorithms.
*/
template<
    typename... NoDataPolicies>
class InputNoDataPolicies:
    public std::tuple<NoDataPolicies...>
{

public:

                   InputNoDataPolicies (NoDataPolicies const&... policies);

                   InputNoDataPolicies (NoDataPolicies&&... policies);

                   ~InputNoDataPolicies()=default;

protected:

};


/*!
    @brief      Constructor.
*/
template<
    typename... NoDataPolicies>
inline InputNoDataPolicies<NoDataPolicies...>::InputNoDataPolicies(
    NoDataPolicies const&... policies)

    : std::tuple<NoDataPolicies...>(policies...)

{
}


/*!
    @brief      Move-constructor.
*/
template<
    typename... NoDataPolicies>
inline InputNoDataPolicies<NoDataPolicies...>::InputNoDataPolicies(
    NoDataPolicies&&... policies)

    : std::tuple<NoDataPolicies...>(std::forward<NoDataPolicies>(policies)...)

{
}

} // namespace algorithm
} // namespace fern
