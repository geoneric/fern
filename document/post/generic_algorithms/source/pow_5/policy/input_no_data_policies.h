#pragma once
#include <tuple>
#include <utility>


template<
    typename... NoDataPolicies>
class InputNoDataPolicies:
    public std::tuple<NoDataPolicies...>
{

public:

                   InputNoDataPolicies (NoDataPolicies const&... policies);

                   InputNoDataPolicies (NoDataPolicies&&... policies);

                   ~InputNoDataPolicies()=default;

};


template<
    typename... NoDataPolicies>
inline InputNoDataPolicies<NoDataPolicies...>::InputNoDataPolicies(
    NoDataPolicies const&... policies)

    : std::tuple<NoDataPolicies...>(policies...)

{
}


template<
    typename... NoDataPolicies>
inline InputNoDataPolicies<NoDataPolicies...>::InputNoDataPolicies(
    NoDataPolicies&&... policies)

    : std::tuple<NoDataPolicies...>(std::forward<NoDataPolicies>(policies)...)

{
}
