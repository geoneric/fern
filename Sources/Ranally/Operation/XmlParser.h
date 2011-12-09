#ifndef INCLUDED_RANALLY_OPERATION_XMLPARSER
#define INCLUDED_RANALLY_OPERATION_XMLPARSER

#include <unicode/unistr.h>
#include <boost/noncopyable.hpp>
#include "Ranally/Operation/Operations.h"



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

  OperationsPtr const& parse           (std::istream& stream) const;

  OperationsPtr const& parse           (UnicodeString const& xml) const;

private:

};

} // namespace operation
} // namespace ranally

#endif
