#pragma once
#include <boost/spirit/home/x3.hpp>
// #include <boost/spirit/home/x3/directive.hpp> // no_case


namespace fern {
namespace wkt {


// CRS WKT characters.
// TODO '(' is also allowed.
static auto const left_delimiter =
    boost::spirit::x3::char_('[')
    ;

// TODO ')' is also allowed.
static auto const right_delimiter =
    boost::spirit::x3::char_(']')
    ;

static auto const wkt_separator =
    boost::spirit::x3::char_(',')
    ;

static auto const double_quote =
    boost::spirit::x3::char_('"')
    ;

static auto const non_double_quote =
    ~double_quote
    ;

static auto const double_quote_symbol =
    double_quote >>
    double_quote
    ;

static auto const quoted_latin_text =
    double_quote >>
    boost::spirit::x3::char_ >>
    double_quote
    ;

} // namespace fern
} // namespace wkt
