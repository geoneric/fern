#pragma once
#include "Ranally/Language/Operation/DataTypes.h"
#include "Ranally/Language/Operation/ValueTypes.h"


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

                   Parameter           ();

                   Parameter           (Parameter const&)=delete;

    Parameter&     operator=           (Parameter const&)=delete;

    virtual        ~Parameter          ();

    ValueTypes     valueTypes          () const;

    DataTypes      dataTypes           () const;

private:

    ValueTypes     _valueTypes;

    DataTypes      _dataTypes;

};

} // namespace ranally
