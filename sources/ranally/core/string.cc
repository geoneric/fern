#include "ranally/core/string.h"
#include <memory>
#include <unicode/ustring.h>


namespace {

//! Encodes a copy of \a string using UTF8 encoding and returns the result.
/*!
  \param     string Unicode string to encode.
  \return    A copy of \a string encoded in UTF8.
*/
std::string encode_in_utf8(
    UnicodeString const& string)
{
    std::string result;

    if(!string.isEmpty()) {
        // At most 4 bytes are needed per Unicode character when encoded in
        // UTF-8.
        unsigned int nrCodePoints = string.countChar32();
        unsigned int maxNrBytesNeeded = 4 * nrCodePoints;
        std::unique_ptr<char> encodedString(new char[maxNrBytesNeeded]);

        // Convert UnicodeString encoded in UTF-16 to UTF-8.
        UErrorCode status = U_ZERO_ERROR;
        int32_t nrCodeUnitsWritten = 0;

        u_strToUTF8(encodedString.get(), maxNrBytesNeeded, &nrCodeUnitsWritten,
            string.getBuffer(), string.length(), &status);
        assert(U_SUCCESS(status));
        assert(nrCodeUnitsWritten > 0);
        assert(static_cast<unsigned int>(nrCodeUnitsWritten) >= nrCodePoints);

        result = std::string(encodedString.get(), nrCodeUnitsWritten);
    }

    return result;
}


//! Decodes \a string from UTF8 encoding and returns the result.
/*!
  \param     string Array of Unicode characters encoded in UTF8.
  \return    Unicode string
*/
UnicodeString decodeFromUTF8(
    std::string const& string)
{
    return UnicodeString(string.c_str(), "UTF-8");
}

} // Anonymous namespace


namespace ranally {

String::String(
    char const* string)

    : UnicodeString(decodeFromUTF8(std::string(string)))

{
}


String::String(
    std::string const& string)

    : UnicodeString(decodeFromUTF8(string))

{
}


String::String(
    UnicodeString const& string)

    : UnicodeString(string)

{
}


String::String(
    boost::format const& format)

    : UnicodeString(decodeFromUTF8(format.str()))

{
}


std::string String::encode_in_utf8() const
{
    return ::encode_in_utf8(*this);
}


bool String::is_empty() const
{
    return UnicodeString::isEmpty();
}


bool String::operator<(
    String const& string) const
{
    return UnicodeString::operator<(string);
}


bool String::operator==(
    String const& string) const
{
    return UnicodeString::operator==(string);
}


String& String::operator+=(
    String const& string)
{
    UnicodeString::operator+=(string);
    return *this;
}


bool String::ends_with(
    String const& string) const
{
    return UnicodeString::endsWith(string);
}


String& String::strip_begin(
    String const& characters)
{
    int32_t index = 0;

    while(index < length() && characters.indexOf(charAt(index)) != -1) {
        ++index;
    }

    assert(index >= 0);
    assert(index <= length());
    remove(0, index);

    return *this;
}


String& String::strip_end(
    String const& characters)
{
    int32_t index = length() - 1;

    while(index >= 0 && characters.indexOf(charAt(index)) != -1) {
        --index;
    }

    assert(index >= 0);
    assert(index < length());
    remove(index + 1, length());

    return *this;
}


//! Trim characters from the start and end of the string.
/*!
  \param     characters String with characters to trim. If empty, whitespace
             characters are trimmed.
  \return    Reference to this.
*/
String& String::strip(
    String const& characters)
{
    if(characters.is_empty()) {
        UnicodeString::trim();
    }
    else {
        strip_begin(characters);
        strip_end(characters);
    }
    return *this;
}


String& String::replace(
    String const& old_string,
    String const& new_string)
{
    findAndReplace(old_string, new_string);
    return *this;
}


String operator+(
    String const& lhs,
    String const& rhs)
{
    String string(lhs);
    string += rhs;
    return string;
}


//! Output operator for String's.
/*!
  The operator in this namespace makes it possible to write
  \code
  BOOST_CHECK_EQUAL(unicodeString1, unicodeString2);
  \endcode
  and get a print of the strings when the test fails.
*/
std::ostream& operator<<(
    std::ostream& stream,
    String const& string)
{
    stream << string.encode_in_utf8();
    return stream;
}

} // namespace ranally
