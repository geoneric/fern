#pragma once
#include <boost/noncopyable.hpp>
#include "Ranally/Operation/Operations.h"
#include "Ranally/Util/String.h"


namespace ranally {
namespace operation {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class XmlParser:
    private boost::noncopyable
{

    friend class XmlParserTest;

public:

                   XmlParser           ();

                   ~XmlParser          ();

    OperationsPtr  parse               (std::istream& stream) const;

    OperationsPtr  parse               (String const& xml) const;

private:

};

} // namespace operation
} // namespace ranally
