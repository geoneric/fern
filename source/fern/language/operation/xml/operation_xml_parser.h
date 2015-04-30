// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include "fern/core/string.h"
#include "fern/language/operation/core/operations.h"


namespace fern {
namespace language {

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

                   ~OperationXmlParser ();

                   OperationXmlParser  (OperationXmlParser&&)=delete;

    OperationXmlParser& operator=      (OperationXmlParser&&)=delete;

                   OperationXmlParser  (OperationXmlParser const&)=delete;

    OperationXmlParser& operator=      (OperationXmlParser const&)=delete;

    OperationsPtr  parse               (std::istream& stream) const;

    OperationsPtr  parse               (String const& xml) const;

private:

};

} // namespace language
} // namespace fern
