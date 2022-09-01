// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/core/string.h"
#include <boost/algorithm/string/join.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/lexical_cast.hpp>
#include <algorithm>
#include <cassert>
#include <iterator>
#include <regex>


namespace fern {

// Encode a copy of @a string using the current encoding and return the result.
// /*!
//   param     string Unicode string to encode.
//   return    A copy of @a string encoded in the current encoding.
//   todo      Implement.
// */
// std::string encode_in_default_encoding(
//     std::string const& string)
// {
//     return string;
// }


// //! Return string decoded from platform's default codepage.
// /*!
//   \param     string String to copy into the new string, encoded in platform's
//              default codepage.
//   \todo      Implement.
// */
// std::string decode_from_default_encoding(
//     std::string const& string)
// {
//     return string;
// }


template<
    class T>
bool is_convertable_to(
    std::string const& string)
{
   bool result = false;

   try {
       boost::lexical_cast<T>(string);
       result = true;
   }
   catch(boost::bad_lexical_cast const&) {
   }

   return result;
}


template<
    class T>
T as(
    std::string const& string)
{
    return is_convertable_to<T>(string)
        ? boost::lexical_cast<T>(string)
        : T();
}


template bool is_convertable_to<double>(std::string const&);
template bool is_convertable_to<int64_t>(std::string const&);
template double as<double>(std::string const&);
template int64_t as<int64_t>(std::string const&);


/*!
    @ingroup    fern_core_group
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
        [](std::string const& string_) { return !string_.empty(); });

    return words;
}


/*!
    @ingroup    fern_core_group
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
    @ingroup    fern_core_group
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
    @ingroup    fern_core_group
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
    @ingroup    fern_core_group
    @brief      Join all @a strings, inserting @a separator inbetween
                the strings, and return the result.

    Comparable to this Python code:
    @code
    separator.join(strings)
    @endcode
*/
std::string join(
    std::vector<std::string> const& strings,
    std::string const& separator)
{
    return boost::algorithm::join(strings, separator);
}


/*!
    @ingroup    fern_core_group
    @brief      Return whether @a string contains @a sub_string.
*/
bool contains(
    std::string const& string,
    std::string const& sub_string)
{
    return string.find(sub_string) != std::string::npos;
}


/*!
    @ingroup    fern_core_group
    @brief      Return whether @a string starts with @a sub_string.
*/
bool starts_with(
    std::string const& string,
    std::string const& sub_string)
{
    return boost::algorithm::starts_with(string, sub_string);
}


/*!
    @ingroup    fern_core_group
    @brief      Return whether @a string ends with @a sub_string.
*/
bool ends_with(
    std::string const& string,
    std::string const& sub_string)
{
    return boost::algorithm::ends_with(string, sub_string);
}


/*!
    @ingroup    fern_core_group
    @brief      Replace all occurences in @a string of @a old_string with
                @a new_string and return a reference to @a string.
    @param      old_string String to search for.
    @param      new_string String to insert if @a old_string is found.
    @return     Reference to @a string
    @warning    If, after replacing @a old_string by @a new_string,
                a new occurence of @a old_string is introduced, this
                occurence is not replaced. For example, if you want to
                replace all occurences of two slasheѕ by one slash, then
                the string a///b is updated to a//b, not a/b. Use a loop
                and contains(std::string const&, std::string const&) if you
                want to be sure that the result doesn't contain an
                occurence of @a old_string.
*/
std::string& replace(
    std::string& string,
    std::string const& old_string,
    std::string const& new_string)
{
    boost::replace_all(string, old_string, new_string);
    return string;
}

} // namespace fern
