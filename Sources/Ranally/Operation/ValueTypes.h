#pragma once
#include <bitset>
#include "Ranally/Operation/ValueType.h"


namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class ValueTypes:
    public std::bitset<NrValueTypes>
{

    friend class ValueTypesTest;

public:

                   ValueTypes          ();

                   ~ValueTypes         ();

private:

};

} // namespace ranally
