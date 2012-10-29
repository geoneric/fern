#pragma once
#include "Ranally/Operation/DataType.h"
#include "Ranally/Operation/ValueType.h"
#include "Ranally/Util/String.h"


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

                   Parameter           (String const& name,
                                        String const& description,
                                        DataTypes dataTypes,
                                        ValueTypes valueTypes);

                   Parameter           (Parameter const& other);

  Parameter&       operator=           (Parameter const& other);

                   ~Parameter          ();

  String const&    name                () const;

  String const&    description         () const;

  DataTypes        dataTypes           () const;

  ValueTypes       valueTypes          () const;

private:

  String           _name;

  String           _description;

  DataTypes        _dataTypes;

  ValueTypes       _valueTypes;

};

} // namespace operation
} // namespace ranally
