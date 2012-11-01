#pragma once
#include "Ranally/Operation/DataType.h"
#include "Ranally/Operation/ValueType.h"
#include "Ranally/Util/String.h"


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

                   Result              (Result const& other);

    Result&        operator=           (Result const& other);

                   ~Result             ();

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
