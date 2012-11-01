#pragma once
#include <bitset>
#include "Ranally/Operation/DataType.h"


namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class DataTypes:
    public std::bitset<NrDataTypes>
{

    friend class DataTypesTest;

public:

                   DataTypes           ();

                   ~DataTypes          ();

private:

};

} // namespace ranally
