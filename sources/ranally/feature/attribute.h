#pragma once
#include "ranally/core/string.h"
#include "ranally/operation/data_type.h"
#include "ranally/operation/value_type.h"


namespace ranally {

// class Feature;

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class Attribute
{

public:

    DataType       data_type           () const;

    ValueType      value_type          () const;

protected:

                   Attribute           (DataType data_type,
                                        ValueType value_type);

                   Attribute           (Attribute const&)=delete;

    Attribute&     operator=           (Attribute const&)=delete;

                   Attribute           (Attribute&&)=delete;

    Attribute&     operator=           (Attribute&&)=delete;

    virtual        ~Attribute          ()=default;

private:

    DataType       _data_type;

    ValueType      _value_type;

};

} // namespace ranally
