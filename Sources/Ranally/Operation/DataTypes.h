#ifndef INCLUDED_RANALLY_OPERATION_DATATYPES
#define INCLUDED_RANALLY_OPERATION_DATATYPES

#include <bitset>
#include "Ranally/Operation/DataType.h"



namespace ranally {
namespace operation {

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

} // namespace operation
} // namespace ranally

#endif
