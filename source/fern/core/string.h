#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <boost/format.hpp>
#include "fern/configure.h"


namespace fern {

//! Unicode string class.
/*!
  This class supports encoding and decoding strings using two encodings: UTF8
  and the default encoding. This latter encoding is the current configured
  platform encoding. UTF8 should be used for strings that are passed around
  internally. The default encoding must be used when string data is obtained
  from or returned to the external context of the software (eg: standard
  output and input streams).

  \todo When converting to / from default encoding, the character type
        depends on the OS. char for ISO/IEC 9945, wchar_t for Windows
        Compare with boost::filesystem's path. Add compile-time checks for
        this.
*/
class String:
    public std::string
{

    friend class DataName;

public:

    static String  decode_from_default_encoding(
                                        char const* string);

    static String  decode_from_default_encoding(
                                        std::string const& string);

    static String  decode_from_default_encoding(
                                        std::wstring const& string);

                   String              ()=default;

                   String              (char const* string);

                   String              (std::string const& string);

                   String              (boost::format const& format);

                   String              (String&&)=default;

    String&        operator=           (String&&)=default;

                   String              (String const&)=default;

    String&        operator=           (String const&)=default;

                   ~String             ()=default;

    bool           operator<           (String const& string) const;

    bool           operator==          (String const& string) const;

    bool           operator!=          (String const& string) const;

    String&        operator+=          (String const& string);

    std::string    encode_in_utf8      () const;

    std::string    encode_in_default_encoding() const;

    bool           is_empty            () const;

    bool           starts_with         (String const& string) const;

    bool           ends_with           (String const& string) const;

    void           clear               ();

    String&        strip               (String const& characters=String());

    bool           contains            (String const& string) const;

    String&        replace             (String const& old_string,
                                        String const& new_string);

#ifndef FERN_COMPILER_DOES_NOT_HAVE_REGEX
    std::vector<String> split          (String characters=String()) const;
#endif

    template<
        class T>
    bool           is_convertable_to   () const;

    template<
        class T>
    T              as                  () const;

private:

    // WARNING
    // std::string is not meant to be inherited from. It doesn't have any
    // virtual functions. Don't add data members here. Our destructor will
    // never be called to destruct them.

    using Base = std::string;

    String&        strip_begin         (String const& characters);

    String&        strip_end           (String const& characters);

};


String             operator+           (String const& lhs,
                                        String const& rhs);

std::ostream&      operator<<          (std::ostream& stream,
                                        String const& string);

String             join                (std::vector<String> const& strings,
                                        String const& separator);

} // namespace fern
