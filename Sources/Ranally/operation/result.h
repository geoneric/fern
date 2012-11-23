#pragma once
#include "ranally/operation/data_type.h"
#include "ranally/operation/value_type.h"
#include "ranally/util/string.h"


namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class Result
{

    friend class ResultTest;

public:

                   Result              (String const& name,
                                        String const& description,
                                        DataType const& dataType,
                                        ValueType const& valueType);

                   ~Result             ()=default;

                   Result              (Result&& other);

    Result&        operator=           (Result&& other);

                   Result              (Result const& other);

    Result&        operator=           (Result const& other);

    String const&  name                () const;

    String const&  description         () const;

    DataType       dataType            () const;

    ValueType      valueType           () const;

private:

    String         _name;

    String         _description;

    DataType       _dataType;

    ValueType      _valueType;

};

} // namespace ranally
