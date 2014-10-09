#pragma once


namespace fern {
namespace algorithm {

/*!
    @ingroup    fern_algorithm_policy_group
    @brief      Domain policy which discards out-of-domain values.

    Use this class if you don't need to test an algorithm's arguments for
    being out of domain. This can be because you are certain the argument
    values are within the algorithm's domain and you don't want to spend
    processor cycles testing values anyway. Or maybe the algorithm just
    accepts all values being passed to it, like default addition.
*/
template<
    typename... Parameters>
class DiscardDomainErrors
{

public:

    static constexpr bool
                   within_domain       (Parameters const&... parameters);

};


template<
    typename... Parameters>
inline constexpr bool DiscardDomainErrors<Parameters...>::within_domain(
    Parameters const&... /* parameters */)
{
    return true;
}


namespace nullary {

using DiscardDomainErrors = DiscardDomainErrors<>;

} // namespace nullary


namespace unary {

template<
    typename Value>
using DiscardDomainErrors = DiscardDomainErrors<Value>;

} // namespace binary


namespace binary {

template<
    typename Value1,
    typename Value2>
using DiscardDomainErrors = DiscardDomainErrors<Value1, Value2>;

} // namespace binary
} // namespace algorithm
} // namespace fern