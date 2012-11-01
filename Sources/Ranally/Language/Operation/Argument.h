#pragma once
#include <boost/noncopyable.hpp>
#include "Ranally/Language/Operation/Parameter.h"


namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class Argument:
    public Parameter
{

    friend class ArgumentTest;

public:

                   Argument            ();

                   ~Argument           ();

protected:

private:

};

} // namespace ranally
