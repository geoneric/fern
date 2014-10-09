#pragma once
#include "fern/core/string.h"
#include "fern/operation/core/data_type.h"
#include "fern/operation/core/value_type.h"


namespace fern {

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

} // namespace fern
