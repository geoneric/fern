#pragma once
#include "geoneric/core/string.h"
#include "geoneric/operation/core/result_type.h"


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
                                        ResultType const& result_type);

                   ~Result             ()=default;

                   Result              (Result&& other);

    Result&        operator=           (Result&& other);

                   Result              (Result const& other);

    Result&        operator=           (Result const& other);

    String const&  name                () const;

    String const&  description         () const;

    ResultType const& result_type      () const;

private:

    String         _name;

    String         _description;

    ResultType     _result_type;

};

} // namespace geoneric
