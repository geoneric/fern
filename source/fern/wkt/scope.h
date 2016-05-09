#pragma once
#include "fern/wkt/core.h"


namespace fern {
namespace wkt {

static auto const scope =
    // boost::spirit::x3::no_case["SCOPE"] >>
    boost::spirit::x3::string("SCOPE") >>
    left_delimiter >>
    quoted_latin_text >>
    right_delimiter
    ;

} // namespace fern
} // namespace wkt
