// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <string>
#include <vector>
#include "fern/configure.h"


namespace fern {

//   TODO  When converting to / from default encoding, the character type
//         depends on the OS. char for ISO/IEC 9945, wchar_t for Windows
//         Compare with boost::filesystem's path. Add compile-time checks for
//         this.

template<
    class T>
bool               is_convertable_to   (std::string const& string);

template<
    class T>
T                  as                  (std::string const& string);


// std::string        decode_from_default_encoding(
//                                         std::string const& string);

// std::string        encode_in_default_encoding(
//                                         std::string const& string);


#ifndef FERN_COMPILER_DOES_NOT_HAVE_REGEX
std::vector<std::string>
                   split               (std::string const& string,
                                        std::string characters=std::string());
#endif

std::string&       strip               (std::string& string,
                                        std::string const& characters=
                                            std::string());

std::string&       strip_begin         (std::string& string,
                                        std::string const& characters);

std::string&       strip_end           (std::string& string,
                                        std::string const& characters);

std::string        join                (std::vector<std::string> const& strings,
                                        std::string const& separator);

bool               contains            (std::string const& string,
                                        std::string const& sub_string);

bool               starts_with         (std::string const& string,
                                        std::string const& sub_string);

bool               ends_with           (std::string const& string,
                                        std::string const& sub_string);

std::string&       replace             (std::string& string,
                                        std::string const& old_string,
                                        std::string const& new_string);

} // namespace fern
