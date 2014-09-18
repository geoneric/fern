#pragma once
#include <tuple>
#include <utility>


namespace fern {

//! Base class for storing input no-data policies of algorithm arguments.
/*!
    \tparam    NoDataPolicies For each input no-data policy its class.

    Most algorithms don't need to be able to test whether elements from their
    arguments have a valid value or not. In these cases the algorithm's result
    can be preprocessed to contain the union of the no-data elements.

    But some algorithms, like cover, do need to be able to test whether
    elements from their arguments have a valid value or not. In these cases
    the input no-data policy passed to the algorithms must contain input
    no-data policies for each argument. Input no-data policy classes that
    inherit from this class can store and retrieve this information. It is
    allowed to pass an empty set of classes in \a NoDataPolicies. That way,
    the single input no-data policy class can be used for both kinds of
    algorithms.
*/
template<
    class... NoDataPolicies>
class InputNoDataPolicies:
    private std::tuple<NoDataPolicies...>
{

public:

    template<
        size_t index>
    auto const&    get                 () const;

protected:

                   InputNoDataPolicies (NoDataPolicies&&... policies);

};


//! Move-constructor.
/*!
*/
template<
    class... NoDataPolicies>
inline InputNoDataPolicies<NoDataPolicies...>::InputNoDataPolicies(
    NoDataPolicies&&... policies)

    : std::tuple<NoDataPolicies...>(std::forward<NoDataPolicies>(policies)...)

{
}


//! Return the input no-data policy for the \a index -th argument.
/*!
*/
template<
    class... NoDataPolicies>
template<
    size_t index>
inline auto const& InputNoDataPolicies<NoDataPolicies...>::get() const
{
    return std::get<index>(dynamic_cast<
        std::tuple<NoDataPolicies...> const&>(*this));
}

} // namespace fern
