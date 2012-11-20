#include "Ranally/Util/string.h"
#include <memory>
#include <unicode/ustring.h>


namespace {

//! Encodes a copy of \a string using UTF8 encoding and returns the result.
/*!
  \param     string Unicode string to encode.
  \return    A copy of \a string encoded in UTF8.
*/
std::string encodeInUTF8(
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


std::string String::encodeInUTF8() const
{
    return ::encodeInUTF8(*this);
}

} // namespace ranally
