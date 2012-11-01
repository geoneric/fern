#pragma once
#include "Ranally/Operation/Operations.h"
#include "Ranally/Util/String.h"


namespace ranally {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class OperationXmlParser
{

    friend class OperationXmlParserTest;

public:

                   OperationXmlParser  ();

                   OperationXmlParser  (OperationXmlParser const&)=delete;

    OperationXmlParser&     operator=  (OperationXmlParser const&)=delete;

                   ~OperationXmlParser ();

    OperationsPtr  parse               (std::istream& stream) const;

    OperationsPtr  parse               (String const& xml) const;

private:

};

} // namespace ranally
