#pragma once
#include "fern/operation/core/argument_type.h"


namespace fern {

//! Base class for values that are handled by the interpreter.
/*!
  These values represent all possible value types that are passed in the
  scripting language.
*/
class Argument
{

public:

    ArgumentType   argument_type       () const;

protected:

                   Argument            (ArgumentType argument_type);

    virtual        ~Argument           ()=default;

                   Argument            (Argument&&)=delete;

    Argument&      operator=           (Argument&&)=delete;

                   Argument            (Argument const&)=delete;

    Argument&      operator=           (Argument const&)=delete;

private:

    ArgumentType   _argument_type;

};

} // namespace fern