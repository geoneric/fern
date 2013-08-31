#pragma once
#include "ranally/core/string.h"
#include "ranally/operation/core/data_types.h"
#include "ranally/operation/core/value_types.h"


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
                                        DataTypes data_types,
                                        ValueTypes value_types);

                   ~Parameter          ();

                   Parameter           (Parameter&& other);

    Parameter&     operator=           (Parameter&& other);

                   Parameter           (Parameter const& other);

    Parameter&     operator=           (Parameter const& other);

    String const&  name                () const;

    String const&  description         () const;

    DataTypes      data_types          () const;

    ValueTypes     value_types         () const;

private:

    String         _name;

    String         _description;

    DataTypes      _data_types;

    ValueTypes     _value_types;

};

} // namespace ranally
