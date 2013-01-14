#pragma once
#include "ranally/feature/value.h"


namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
template<class T>
class ScalarValue:
    public Value
{

    friend class ScalarValueTest;

public:

                   ScalarValue         (T const& value);

                   ScalarValue         (ScalarValue const&)=delete;

    ScalarValue&   operator=           (ScalarValue const&)=delete;

                   ScalarValue         (ScalarValue&&)=delete;

    ScalarValue&   operator=           (ScalarValue&&)=delete;

                   ~ScalarValue        ();

protected:

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

} // namespace ranally
