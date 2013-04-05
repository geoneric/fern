#pragma once
#include "ranally/interpreter/value_type.h"


namespace ranally {
namespace interpreter {

//! Base class for values that are handled by the interpreter.
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class Value
{

public:

    ValueType      value_type          () const;

protected:

                   Value               (ValueType value_type);

    virtual        ~Value              ()=default;

                   Value               (Value&&)=delete;

    Value&         operator=           (Value&&)=delete;

                   Value               (Value const&)=delete;

    Value&         operator=           (Value const&)=delete;

private:

    ValueType      _value_type;

};

} // namespace interpreter
} // namespace ranally
