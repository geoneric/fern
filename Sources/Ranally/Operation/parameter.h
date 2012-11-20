#pragma once
#include "Ranally/Operation/data_type.h"
#include "Ranally/Operation/value_type.h"
#include "Ranally/Util/string.h"


namespace ranally {

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

                   ~Parameter          ();

                   Parameter           (Parameter&& other);

    Parameter&     operator=           (Parameter&& other);

                   Parameter           (Parameter const& other);

    Parameter&     operator=           (Parameter const& other);

    String const&  name                () const;

    String const&  description         () const;

    DataTypes      dataTypes           () const;

    ValueTypes     valueTypes          () const;

private:

    String         _name;

    String         _description;

    DataTypes      _dataTypes;

    ValueTypes     _valueTypes;

};

} // namespace ranally
