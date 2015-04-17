// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/core/string.h"
#include <algorithm>
#include <cassert>
#include <iterator>
#include <regex>
#include <boost/algorithm/string/join.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/lexical_cast.hpp>


//! Encode a copy of \a string using UTF8 encoding and return the result.
/*!
  \param     string Unicode string to encode.
  \return    A copy of \a string encoded in UTF8.
  \todo      Implement.
*/
std::string encode_in_utf8(
    std::string const& string)
{
    return string;
}


//! Encode a copy of \a string using the current encoding and return the result.
/*!
  \param     string Unicode string to encode.
  \return    A copy of \a string encoded in the current encoding.
  \todo      Implement.
*/
std::string encode_in_default_encoding(
    std::string const& string)
{
    return string;
}


namespace fern {

//! Return string decoded from platform's default codepage.
/*!
  \param     string String to copy into the new string, incoded in platform's
             default codepage.
  \return    New String instance.
  \todo      Implement.
*/
String String::decode_from_default_encoding(
    char const* string)
{
    return String(string);
}


//! Return string decoded from platform's default codepage.
/*!
  \param     string String to copy into the new string, incoded in platform's
             default codepage.
  \return    New String instance.
  \todo      Implement.
*/
String String::decode_from_default_encoding(
    std::string const& string)
{
    return String(string);
}


//! Return string decoded from platform's default codepage.
/*!
  \param     string String to copy into the new string, incoded in platform's
             default codepage.
  \return    New String instance.
  \todo      Implement.
*/
String String::decode_from_default_encoding(
    std::wstring const& /* string */)
{
    assert(false);
    return String("");
}


//! Constructor.
/*!
  \param     string String to copy into the new string, encoded in UTF8.
*/
String::String(
    char const* string)

    : Base(string)

{
}


//! Constructor.
/*!
  \param     string String to copy into the new string, encoded in UTF8.
*/
String::String(
    std::string const& string)

    : Base(string)

{
}


//! Constructor.
/*!
  \param     format Format containing string to copy into the new string,
             encoded in UTF8.
*/
String::String(
    boost::format const& format)

    : Base(format.str())

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
    return Base::empty();
}


bool String::operator<(
    String const& string) const
{
    return static_cast<std::string const&>(*this) <
        static_cast<std::string const&>(string);
}


bool String::operator==(
    String const& string) const
{
    return static_cast<std::string const&>(*this) ==
        static_cast<std::string const&>(string);
}


bool String::operator!=(
    String const& string) const
{
    return static_cast<std::string const&>(*this) !=
        static_cast<std::string const&>(string);
}


String& String::operator+=(
    String const& string)
{
    Base::operator+=(string);
    return *this;
}


//! Return whether this string starts with \a string.
/*!
  \param     string String to compare.
*/
bool String::starts_with(
    String const& string) const
{
    return boost::algorithm::starts_with(*this, string);
}


//! Return whether this string ends with \a string.
/*!
  \param     string String to compare.
*/
bool String::ends_with(
    String const& string) const
{
    return boost::algorithm::ends_with(*this, string);
}


void String::clear()
{
    Base::clear();
}


String& String::strip(
    String const& characters)
{
    fern::strip(*this, characters);
    return *this;
}


//! Return whether this string contains \a string.
/*!
  \param     string String to search for.
*/
bool String::contains(
    String const& string) const
{
    return find(string) != std::string::npos;
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
    boost::replace_all(static_cast<Base&>(*this),
        static_cast<Base const&>(old_string), static_cast<Base const&>(new_string));
    return *this;
}


template<
    class T>
bool String::is_convertable_to() const
{
    bool result = false;

    try {
        boost::lexical_cast<T>(*this);
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
        ? boost::lexical_cast<T>(*this)
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


/*!
    @brief      Split @a string by @a characters and return the result.
    @param      characters Characters to split string by. If this string
                is empty, the string is split by whitespace characters.
    @return     Vector with strings.
    @warning    The @a characters passed in are used in a regular
                expression, where the string of characters is put in
                a bracket expression. It is assumed this results in a
                valid regular expression.
*/
std::vector<std::string> split(
    std::string const& string,
    std::string characters)
{
    if(characters.empty()) {
        characters = "[:space:]";
    }

    std::regex regular_expression(
        std::string("[") + characters + std::string("]+"));
    std::vector<std::string> words;

    std::copy_if(std::sregex_token_iterator(string.begin(), string.end(),
        regular_expression, -1), std::sregex_token_iterator(),
        std::back_inserter(words),
        [](std::string const& string) { return !string.empty(); });

    // std::vector<std::string> words2;
    // std::copy(words1.begin(), words1.end(), std::back_inserter(words2));

    return words;
}


/*!
    @brief      Strip @a characters from the start and end of @a string and
                return the result.
    @param      string String to trim. A reference to this string is
                returned.
    @param      characters String with characters to strip. If empty,
                whitespace characters are trimmed.
*/
std::string& strip(
    std::string& string,
    std::string const& characters)
{
    if(characters.empty()) {
        boost::algorithm::trim(string);
    }
    else {
        strip_begin(string, characters);
        strip_end(string, characters);
    }

    return string;
}


/*!
    @brief      Strip @a characters from the begin of @a string and return
                a reference to the updated @a string.
*/
std::string& strip_begin(
    std::string& string,
    std::string const& characters)
{
    boost::trim_left_if(string, [&](char c){
        return characters.find(c) != std::string::npos; });
    return string;
}


/*!
    @brief      Strip @a characters from the end of @a string and return
                a reference to the updated @a string.
*/
std::string& strip_end(
    std::string& string,
    std::string const& characters)
{
    boost::trim_right_if(string, [&](char c){
        return characters.find(c) != std::string::npos; });
    return string;
}


/*!
    @brief      Join all @a strings, inserting @a separator inbetween
                the strings, and return the result.

    Comparable to this Python code:
    @code
    separator.join(strings)
    @endcode
*/
String join(
    std::vector<String> const& strings,
    String const& separator)
{
    std::vector<std::string> words;
    std::copy(strings.begin(), strings.end(), std::back_inserter(words));

    return boost::algorithm::join(words, static_cast<std::string const&>(
        separator));
}


// #ifndef FERN_COMPILER_DOES_NOT_HAVE_REGEX
// /*!
//   @brief     Split this string by @a characters and return the result.
//   @sa        fern::split(String const&, String)
// */
// std::vector<String> String::split(
//     String characters) const
// {
//     return fern::split(*this, characters);
// }
// #endif

} // namespace fern
