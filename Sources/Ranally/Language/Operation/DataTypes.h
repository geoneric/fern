#ifndef INCLUDED_RANALLY_LANGUAGE_OPERATION_DATATYPES
#define INCLUDED_RANALLY_LANGUAGE_OPERATION_DATATYPES

#include <bitset>



namespace ranally {
namespace language {
namespace operation {

enum DataType {
  Scalar,
  AsArgument,
  NrDataTypes
};



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

protected:

private:

};

} // namespace operation
} // namespace language
} // namespace ranally

#endif
