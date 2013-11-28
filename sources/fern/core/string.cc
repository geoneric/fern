#include "fern/core/string.h"
#include <memory>
#include <unicode/ustring.h>
#include <unicode/regex.h>
#include <boost/lexical_cast.hpp>


namespace {

//! Encode a copy of \a string using UTF8 encoding and return the result.
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


//! Encode a copy of \a string using the current encoding and return the result.
/*!
  \param     string Unicode string to encode.
  \return    A copy of \a string encoded in the current encoding.
*/
std::string encode_in_default_encoding(
    UnicodeString const& string)
{
    // Number of Unicode characters in the string.
    int32_t nr_code_points = string.countChar32();

    // Number of (UChar) units representing the Unicode characters.
    int32_t nr_code_units = string.length();

    int32_t max_nr_bytes_needed = 4 * nr_code_points;
    std::unique_ptr<char> encoded_string(new char[max_nr_bytes_needed]);

    int32_t nr_bytes_written = string.extract(0, nr_code_units,
        encoded_string.get(), max_nr_bytes_needed);
    assert(nr_bytes_written <= max_nr_bytes_needed);

    return std::string(encoded_string.get(), nr_bytes_written);
}


//! Decode \a string from UTF8 encoding and return the result.
/*!
  \param     string Array of Unicode characters encoded in UTF8.
  \return    Unicode string.
*/
UnicodeString decode_from_utf8(
    std::string const& string)
{
    return UnicodeString(string.c_str(), "UTF-8");
}

} // Anonymous namespace


namespace fern {

//! Return string decoded from platform's default codepage.
/*!
  \param     string String to copy into the new string, incoded in platform's
             default codepage.
  \return    New String instance.
*/
String String::decode_from_default_encoding(
    char const* string)
{
    return String(UnicodeString(string));
}


//! Return string decoded from platform's default codepage.
/*!
  \param     string String to copy into the new string, incoded in platform's
             default codepage.
  \return    New String instance.
*/
String String::decode_from_default_encoding(
    std::string const& string)
{
    return String(UnicodeString(string.c_str()));
}


//! Constructor.
/*!
  \param     string String to copy into the new string, encoded in UTF8.
*/
String::String(
    char const* string)

    : UnicodeString(decode_from_utf8(std::string(string)))

{
}


//! Constructor.
/*!
  \param     string String to copy into the new string, encoded in UTF8.
*/
String::String(
    std::string const& string)

    : UnicodeString(decode_from_utf8(string))

{
}


//! Constructor.
/*!
  \param     string String to copy into the new string.

  This constructor is private because we don't want clients to depend on
  the fact that this class inherits from UnicodeString.
*/
String::String(
    UnicodeString const& string)

    : UnicodeString(string)

{
}


//! Constructor.
/*!
  \param     format Format containing string to copy into the new string,
             encoded in UTF8.
*/
String::String(
    boost::format const& format)

    : UnicodeString(decode_from_utf8(format.str()))

{
}


//! Return a copy of the instance encoded in UTF8.
/*!
  \return    UTF8 encoded copy of this string.
  \sa        encode_in_default_encoding()
*/
std::string String::encode_in_utf8() const
{
    return ::encode_in_utf8(*this);
}


//! Return a copy of the instance encoded in the default encoding.
/*!
  \return    Default encoded copy of this string.
  \sa        encode_in_utf8()
*/
std::string String::encode_in_default_encoding() const
{
    return ::encode_in_default_encoding(*this);
}


//! Return whether the string is empty.
/*!
*/
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


bool String::operator!=(
    String const& string) const
{
    return UnicodeString::operator!=(string);
}


String& String::operator+=(
    String const& string)
{
    UnicodeString::operator+=(string);
    return *this;
}


//! Return whether this string starts with \a string.
/*!
  \param     string String to compare.
*/
bool String::starts_with(
    String const& string) const
{
    return UnicodeString::startsWith(string);
}


//! Return whether this string ends with \a string.
/*!
  \param     string String to compare.
*/
bool String::ends_with(
    String const& string) const
{
    return UnicodeString::endsWith(string);
}


void String::clear()
{
    UnicodeString::remove();
}


//! Strip characters from \a characters from the begin of the string.
/*!
  \param     characters String with characters to strip.
  \return    Reference to *this.
*/
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


//! Strip characters from \a characters from the end of the string.
/*!
  \param     characters String with characters to strip.
  \return    Reference to *this.
*/
String& String::strip_end(
    String const& characters)
{
    int32_t index = length() - 1;

    while(index >= 0 && characters.indexOf(charAt(index)) != -1) {
        --index;
    }

    assert(index >= -1);
    assert(index < length());
    remove(index + 1, length());

    return *this;
}


//! Strip characters from the start and end of the string.
/*!
  \param     characters String with characters to strip. If empty, whitespace
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


//! Return whether this string contains \a string.
/*!
  \param     string String to search for.
*/
bool String::contains(
    String const& string) const
{
    return indexOf(string) != -1;
}


//! Replace all occurences of \a old_string with \a new_string.
/*!
  \param     old_string String to search for.
  \param     new_string String to insert if \a old_string is found.
  \return    Reference to *this.
  \warning   If, after replacing \a old_string by \a new_string, a new
             occurence of \a old_string is introduced, this occurence is not
             replaced. For example, if you want to replace all
             occurences of two slasheѕ by one slash, then the string a///b is
             updated to a//b, not a/b. Use a loop and contains(String const&)
             if you want to be sure that the result doesn't contain an
             occurence of \a old_string.
*/
String& String::replace(
    String const& old_string,
    String const& new_string)
{
    findAndReplace(old_string, new_string);
    return *this;
}


template<
    class T>
bool String::is_convertable_to() const
{
    bool result = false;

    try {
        boost::lexical_cast<T>(encode_in_utf8());
        result = true;
    }
    catch(boost::bad_lexical_cast const&) {
    }

    return result;
}


template<
    class T>
T String::as() const
{
    return is_convertable_to<T>()
        ? boost::lexical_cast<T>(encode_in_utf8())
        : T();
}


template bool String::is_convertable_to<double>() const;
template bool String::is_convertable_to<int64_t>() const;
template double String::as<double>() const;
template int64_t String::as<int64_t>() const;


//! Return the concatenation of \a lhs and \a rhs.
/*!
  \param     lhs Left hand side string.
  \param     rhs Righ hand side string.
  \return    New string.
*/
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
    stream << string.encode_in_default_encoding();
    return stream;
}


//! Join all \a strings, inserting \a separator inbetween the strings, and return the result.
/*!
  \param     strings String to join.
  \param     separator String to insert inbetween the strings.
  \return    New string.

  Comparable to this Python code:
  \code
  separator.join(strings)
  \endcode
*/
String join(
    std::vector<String> const& strings,
    String const& separator)
{
    String result;

    if(!strings.empty()) {
        result += strings.front();

        for(size_t i = 1; i < strings.size(); ++i) {
            result += separator + strings[i];
        }
    }

    return result;
}


//! Split this string by \a characters and return the result.
/*!
  \param     characters Characters to split string by. If this string is empty,
             the string is split by whitespace characters.
  \return    Vector with strings.
  \warning   The \a characters passed in are used in a regular expression,
             where the string of characters is put in a bracket expression.
             It is assumed this results in a valid regular expression.
*/
std::vector<String> String::split(
    String characters) const
{
    String string(*this);
    string.strip(characters);

    UErrorCode status = U_ZERO_ERROR;
    if(characters.is_empty()) {
        characters = "\\s";
    }
    String regex("[" + characters + "]+");
    RegexMatcher matcher(regex, 0, status);
    assert(!U_FAILURE(status));

    int const max_nr_words = 10;
    String words[max_nr_words];
    int nr_words = matcher.split(string, words, max_nr_words, status);
    assert(!U_FAILURE(status));

    return std::vector<String>(words, words + nr_words);
}

} // namespace fern
