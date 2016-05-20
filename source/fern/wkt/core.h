#pragma once
#include <boost/spirit/home/x3.hpp>
// #include <boost/spirit/home/x3/directive.hpp> // no_case


namespace fern {
namespace wkt {


// CRS WKT characters.
// TODO '(' is also allowed.
static auto const left_delimiter =
    boost::spirit::x3::lit('[')
    ;

// TODO ')' is also allowed.
static auto const right_delimiter =
    boost::spirit::x3::lit(']')
    ;

static auto const wkt_separator =
    boost::spirit::x3::lit(',')
    ;

static auto const double_quote =
    boost::spirit::x3::lit('"')
    ;

static auto const non_double_quote =
    ~boost::spirit::x3::char_('"')
    ;

static auto const double_quote_symbol =
    double_quote >>
    double_quote
    ;

static auto const quoted_latin_text =
    boost::spirit::x3::rule<class quoted_latin_text, std::string>() =
        double_quote >>
        // Turn skipper off.
        boost::spirit::x3::lexeme[*non_double_quote] >>
        double_quote
        ;

static auto const unsigned_integer = boost::spirit::x3::uint64;

static auto const signed_integer = boost::spirit::x3::int64;

// static auto const unsigned_float =
//     boost::spirit::x3::rule<class unsigned_float, double>() =
//         ~boost::spirit::x3::lit('-') >> boost::spirit::x3::double_
//         ;

static auto const unsigned_float = boost::spirit::x3::double_;

static auto const signed_float = boost::spirit::x3::double_;

static auto const unsigned_numeric_literal =
    // boost::spirit::x3::rule<class unsigned_numeric_literal, double>() =
        unsigned_float // | unsigned_integer
        ;

static auto const signed_numeric_literal =
    // boost::spirit::x3::rule<class signed_numeric_literal, double>() =
        signed_float // | signed_integer
        ;

static auto const number =
    // boost::spirit::x3::rule<class number, double>() =
        // signed_numeric_literal | unsigned_numeric_literal
        boost::spirit::x3::double_
        ;

} // namespace wkt
} // namespace fern
