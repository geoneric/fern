#ifndef INCLUDED_RANALLY_UTIL_STRING
#define INCLUDED_RANALLY_UTIL_STRING

#include <iostream>
#include <unicode/unistr.h>



namespace ranally {
namespace util {

std::string        encodeInUTF8        (UnicodeString const& string);

UnicodeString      decodeFromUTF8      (std::string const& string);

} // namespace util
} // namespace ranally



namespace boost { namespace test_tools { namespace tt_detail {

//! Output operator for UnicodeString's.
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
         UnicodeString const& string)
{
  stream << ranally::util::encodeInUTF8(string);

  return stream;
}

}}}

#endif
