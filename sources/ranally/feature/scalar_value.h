#pragma once
#include "ranally/feature/value.h"


namespace ranally {

//! ScalarValue is an attribute value that contains a single value.
/*!
  ScalarValue instances can be associated with a ScalarDomain in a
  ScalarAttribute.
*/
template<class T>
class ScalarValue:
    public Value
{

public:

                   ScalarValue         (T const& value);

                   ScalarValue         (ScalarValue const&)=delete;

    ScalarValue&   operator=           (ScalarValue const&)=delete;

                   ScalarValue         (ScalarValue&&)=delete;

    ScalarValue&   operator=           (ScalarValue&&)=delete;

                   ~ScalarValue        ();

    T const&       operator()          () const;

private:

    T              _value;

};


template<class T>
inline ScalarValue<T>::ScalarValue(
    T const& value)

    : Value(),
      _value(value)

{
}


template<class T>
inline ScalarValue<T>::~ScalarValue()
{
}


template<class T>
inline T const& ScalarValue<T>::operator()() const
{
    return _value;
}

} // namespace ranally
