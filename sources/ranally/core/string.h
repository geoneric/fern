#pragma once
#include <iostream>
#include <boost/format.hpp>
#include <unicode/unistr.h>


namespace ranally {

//! Unicode string class.
/*!
*/
// This class privately inherits from UnicodeString so we can keep track of
// the methods that are actually used. If necessary we may want to move to
// another string base type (C++ Unicode string type?!).
class String:
    private UnicodeString
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

                   ~String             ()=default;

    bool           operator<           (String const& string) const;

    bool           operator==          (String const& string) const;

    String&        operator+=          (String const& string);

    std::string    encode_in_utf8      () const;

    bool           is_empty            () const;

    bool           ends_with           (String const& string) const;

    String&        strip               (String const& characters=String());

    String&        replace             (String const& old_string,
                                        String const& new_string);

private:

    String&        strip_begin         (String const& characters);

    String&        strip_end           (String const& characters);

};


String             operator+           (String const& lhs,
                                        String const& rhs);

std::ostream&      operator<<          (std::ostream& stream,
                                        String const& string);

} // namespace ranally
