#pragma once


namespace fern {

//! Domain policy which discards out-of-domain values.
/*!
  \tparam    A1 Type of first argument.
  \tparam    A2 Type of second argument.
*/
template<
    class A1,
    class A2>
class DiscardDomainErrors
{

public:

    static constexpr bool
                   within_domain       (A1 const& argument1,
                                        A2 const& argument2);

protected:

                   DiscardDomainErrors ()=default;

                   DiscardDomainErrors (DiscardDomainErrors&&)=default;

    DiscardDomainErrors&
                   operator=           (DiscardDomainErrors&&)=default;

                   DiscardDomainErrors (DiscardDomainErrors const&)=default;

    DiscardDomainErrors&
                   operator=           (DiscardDomainErrors const&)=default;

                   ~DiscardDomainErrors()=default;

private:

};


//! Check whether \a argument1 and \a argument2 fall within the domain of valid values.
/*!
  \return    true
*/
template<
    class A1,
    class A2>
inline constexpr bool DiscardDomainErrors<A1, A2>::within_domain(
    A1 const& /* argument1 */,
    A2 const& /* argument2 */)
{
    return true;
}

} // namespace fern
