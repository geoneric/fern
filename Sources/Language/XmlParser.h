#ifndef INCLUDED_RANALLY_XMLPARSER
#define INCLUDED_RANALLY_XMLPARSER

#include <unicode/unistr.h>

#include "dev_XercesClient.h"



namespace ranally {

typedef int SyntaxTree;

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class XmlParser: public dev::XercesClient
{

  friend class XmlParserTest;

private:

protected:

public:

                   XmlParser           ();

  /* virtual */    ~XmlParser          ();

  SyntaxTree       parse               (UnicodeString const& xml) const;

  SyntaxTree       parse               (std::istream& stream) const;

};

} // namespace ranally

#endif
