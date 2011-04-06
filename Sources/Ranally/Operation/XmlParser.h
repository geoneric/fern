#ifndef INCLUDED_RANALLY_OPERATION_XMLPARSER
#define INCLUDED_RANALLY_OPERATION_XMLPARSER

#include <map>
#include <unicode/unistr.h>
#include <boost/noncopyable.hpp>
#include "Operation-pskel.hxx"



namespace ranally {
namespace operation {

//! short_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED
/*!
  longer_description_HORRIBLE_LONG_STRING_TO_NOTICE_THAT_IT_SHOULD_BE_REPLACED

  \sa        .
*/
class XmlParser: private boost::noncopyable
{

  friend class XmlParserTest;

public:

                   XmlParser           ();

  /* virtual */    ~XmlParser          ();

   std::map<UnicodeString, Operation_pskel> parse(
                                        UnicodeString const& xml) const;

protected:

private:

};

} // namespace operation
} // namespace ranally

#endif
