#pragma once


namespace fern {

//! Domain policy which discards out-of-domain values.
/*!
  \tparam    Arguments Types of arguments.
*/
template<
    class... Arguments>
class DiscardDomainErrors
{

public:

    // static_assert(std::is_arithmetic<A1>::value, "");
    // static_assert(std::is_arithmetic<A2>::value, "");

    static constexpr bool
                   within_domain       (Arguments const&... arguments);

protected:

                   DiscardDomainErrors ()=default;

                   DiscardDomainErrors (DiscardDomainErrors&&)=default;

                   DiscardDomainErrors (DiscardDomainErrors const&)=default;

    DiscardDomainErrors&
                   operator=           (DiscardDomainErrors&&)=default;

    DiscardDomainErrors&
                   operator=           (DiscardDomainErrors const&)=default;

                   ~DiscardDomainErrors()=default;

private:

};


//! Check whether the arguments passed in fall within the domain of valid values.
/*!
  \return    true
*/
template<
    class... Arguments>
inline constexpr bool DiscardDomainErrors<Arguments...>::within_domain(
    Arguments const&... /* arguments */)
{
    return true;
}

} // namespace fern
