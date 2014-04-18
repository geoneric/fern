#pragma once


namespace fern {

//! Domain policy which discards out-of-domain values.
/*!
    Use this class if you don't need to test an algorithm's arguments for
    being out of domain. This can be because you are certain the argument
    values are within the algorithm's domain and you don't want to spend
    processor cycles testing values anyway. Or maybe the algorithm just
    accepts all values being passed to it, like default addition.
*/
template<
    class... Parameters>
class DiscardDomainErrors
{

public:

    static constexpr bool
                   within_domain       (Parameters const&... parameters);

};


template<
    class... Parameters>
inline constexpr bool DiscardDomainErrors<Parameters...>::within_domain(
    Parameters const&... /* parameters */)
{
    return true;
}


namespace binary {

template<
    class Value1,
    class Value2>
using DiscardDomainErrors = DiscardDomainErrors<Value1, Value2>;

} // namespace binary
} // namespace fern
