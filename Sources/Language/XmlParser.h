#ifndef INCLUDED_RANALLY_XMLPARSER
#define INCLUDED_RANALLY_XMLPARSER

#include <unicode/unistr.h>
#include <boost/shared_ptr.hpp>



namespace ranally {
namespace language {

class ScriptVertex;

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class XmlParser
{

  friend class XmlParserTest;

private:

protected:

public:

                   XmlParser           ();

  /* virtual */    ~XmlParser          ();

  boost::shared_ptr<ScriptVertex> parse(UnicodeString const& xml) const;

  boost::shared_ptr<ScriptVertex> parse(std::istream& stream) const;

};

} // namespace language
} // namespace ranally

#endif
