#pragma once


namespace fern {

template<
    class A1,
    class A2>
class DiscardDomainErrors {

public:

    static bool    within_domain       (A1 const& argument1,
                                        A2 const& argument2);

protected:

                   DiscardDomainErrors ()=default;

                   DiscardDomainErrors (DiscardDomainErrors&&)=delete;

    DiscardDomainErrors&
                   operator=           (DiscardDomainErrors&&)=delete;

                   DiscardDomainErrors (DiscardDomainErrors const&)=delete;

    DiscardDomainErrors&
                   operator=           (DiscardDomainErrors const&)=delete;

                   ~DiscardDomainErrors()=default;

private:

};


template<
    class A1,
    class A2>
inline bool DiscardDomainErrors<A1, A2>::within_domain(
    A1 const& /* argument1 */,
    A2 const& /* argument2 */)
{
    return true;
}

} // namespace fern
