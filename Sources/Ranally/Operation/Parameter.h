#ifndef INCLUDED_RANALLY_OPERATION_PARAMETER
#define INCLUDED_RANALLY_OPERATION_PARAMETER

#include <unicode/unistr.h>
#include "Ranally/Operation/DataType.h"
#include "Ranally/Operation/ValueType.h"



namespace ranally {
namespace operation {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class Parameter
{

  friend class ParameterTest;

public:

                   Parameter           (UnicodeString const& name,
                                        UnicodeString const& description,
                                        DataTypes dataTypes,
                                        ValueTypes valueTypes);

                   Parameter           (Parameter const& other);

  Parameter&       operator=           (Parameter const& other);

                   ~Parameter          ();

  UnicodeString const& name            () const;

  UnicodeString const& description     () const;

  DataTypes        dataTypes           () const;

  ValueTypes       valueTypes          () const;

private:

  UnicodeString    _name;

  UnicodeString    _description;

  DataTypes        _dataTypes;

  ValueTypes       _valueTypes;

};

} // namespace operation
} // namespace ranally

#endif
