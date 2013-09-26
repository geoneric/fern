#pragma once
#include "geoneric/core/data_types.h"
#include "geoneric/core/string.h"
#include "geoneric/core/value_types.h"


namespace geoneric {

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
                                        DataTypes const& data_type,
                                        ValueTypes const& value_type);

                   ~Result             ()=default;

                   Result              (Result&& other);

    Result&        operator=           (Result&& other);

                   Result              (Result const& other);

    Result&        operator=           (Result const& other);

    String const&  name                () const;

    String const&  description         () const;

    DataTypes      data_type           () const;

    ValueTypes     value_type          () const;

private:

    String         _name;

    String         _description;

    DataTypes      _data_type;

    ValueTypes     _value_type;

};

} // namespace geoneric
