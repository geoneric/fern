#ifndef INCLUDED_RANALLY_UTIL_STRING
#define INCLUDED_RANALLY_UTIL_STRING

#include <iostream>
#include <boost/format.hpp>
#include <unicode/unistr.h>



namespace ranally {

class String:
  public UnicodeString
{

private:

public:

                   String              ();

                   String              (char const* string);

                   String              (std::string const& string);

                   String              (UnicodeString const& string);

                   String              (boost::format const& format);

  virtual          ~String             ();

  std::string      encodeInUTF8        () const;

};



//! Output operator for String's.
/*!
  \param     .
  \return    .
  \exception .
  \warning   .
  \sa        .
  \todo      Why doesn't this work when this operator is in the global
             namespace? (Not sure if we want that, but I prefer not to know
             about the boost::test_tools::tt_detail stuff.)

  The operator in this namespace makes it possible to write
  \code
  BOOST_CHECK_EQUAL(unicodeString1, unicodeString2);
  \endcode
  and get a print of the strings when the test fails.
*/
inline std::ostream& operator<<(
  std::ostream& stream,
  String const& string)
{
  stream << string.encodeInUTF8();
  return stream;
}

} // namespace ranally

#endif
