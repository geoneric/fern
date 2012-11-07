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

                   ~DataTypes          ()=default;

                   DataTypes           (DataTypes&&)=delete;

    DataTypes&     operator=           (DataTypes&&)=delete;

                   DataTypes           (DataTypes const&)=delete;

    DataTypes&     operator=           (DataTypes const&)=delete;

private:

};

} // namespace ranally
