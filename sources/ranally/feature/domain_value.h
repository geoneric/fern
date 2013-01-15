#pragma once
#include <vector>
#include "ranally/feature/value.h"


namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
template<class T>
class DomainValue:
    public Value
{

public:

    template<class Domain>
                   DomainValue         (Domain const& domain);

                   DomainValue         (DomainValue const&)=delete;

    DomainValue&   operator=           (DomainValue const&)=delete;

                   DomainValue         (DomainValue&&)=delete;

    DomainValue&   operator=           (DomainValue&&)=delete;

                   ~DomainValue        ();

    size_t         size                () const;

    std::vector<T> const& operator()   () const;

    std::vector<T>& operator()         ();

private:

    size_t         _size;

    std::vector<T> _values;

};


template<class T>
template<class Domain>
inline DomainValue<T>::DomainValue(
    Domain const& domain)

    : Value(),
      _size(domain.geometry().size()),
      _values(_size)

{
}


template<class T>
inline DomainValue<T>::~DomainValue()
{
}


template<class T>
inline size_t DomainValue<T>::size() const
{
    return _size;
}


template<class T>
inline std::vector<T> const& DomainValue<T>::operator()() const
{
    return _values;
}


template<class T>
inline std::vector<T>& DomainValue<T>::operator()()
{
    return _values;
}

} // namespace ranally
