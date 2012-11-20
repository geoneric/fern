#pragma once
#include <bitset>
#include "Ranally/Operation/value_type.h"


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

                   ~ValueTypes         ()=default;

                   ValueTypes          (ValueTypes&&)=delete;

    ValueTypes&    operator=           (ValueTypes&&)=delete;

                   ValueTypes          (ValueTypes const&)=delete;

    ValueTypes&    operator=           (ValueTypes const&)=delete;

private:

};

} // namespace ranally
