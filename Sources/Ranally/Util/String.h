#pragma once
#include <iostream>
#include <boost/format.hpp>
#include <unicode/unistr.h>


namespace ranally {

class String:
    public UnicodeString
{

public:

                   String              ()=default;

                   String              (char const* string);

                   String              (std::string const& string);

                   String              (UnicodeString const& string);

                   String              (boost::format const& format);

                   String              (String&&)=default;

    String&        operator=           (String&&)=default;

                   String              (String const&)=default;

    String&        operator=           (String const&)=default;

    virtual        ~String             ()=default;

    std::string    encodeInUTF8        () const;

private:

};


//! Output operator for String's.
/*!
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
