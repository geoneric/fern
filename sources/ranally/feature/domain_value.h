#pragma once
#include <unordered_map>
#include "ranally/feature/fid_map.h"
#include "ranally/feature/value.h"


namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
template<
    class T>
class DomainValue:
    public Value,
    public FidMap<T>
{

public:

    template<class Domain>
                   DomainValue         (Domain const& domain);

                   DomainValue         (DomainValue const&)=delete;

    DomainValue&   operator=           (DomainValue const&)=delete;

                   DomainValue         (DomainValue&&)=delete;

    DomainValue&   operator=           (DomainValue&&)=delete;

                   ~DomainValue        ();

private:

};


template<
    class T>
template<
    class Domain>
inline DomainValue<T>::DomainValue(
    Domain const& domain)

    : Value(),
      FidMap<T>()

{
    for(typename Domain::value_type const& value: domain) {
        this->insert(value.first, T());
    }
}


template<
    class T>
inline DomainValue<T>::~DomainValue()
{
}

} // namespace ranally
