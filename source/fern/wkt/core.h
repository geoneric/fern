#pragma once
#include <boost/spirit/home/x3.hpp>
// #include <boost/spirit/home/x3/directive.hpp> // no_case


namespace fern {
namespace wkt {
namespace grammar {

template<
    typename T>
auto as = [](auto parser)
{
    return boost::spirit::x3::rule<struct _, T>{} =
        boost::spirit::x3::as_parser(parser);
};


// CRS WKT characters.
// TODO '(' is also allowed.
static auto const left_delimiter =
    boost::spirit::x3::lit('[');

// TODO ')' is also allowed.
static auto const right_delimiter =
    boost::spirit::x3::lit(']');

static auto const wkt_separator =
    boost::spirit::x3::lit(',');

static auto const hyphen =
    boost::spirit::x3::lit('-');

static auto const colon =
    boost::spirit::x3::lit(':');

static auto const period =
    boost::spirit::x3::lit('.');

// static auto const plus_sign =
//     boost::spirit::x3::char_('+');
// 
// static auto const minus_sign =
//     boost::spirit::x3::char_('-');

static auto const double_quote =
    boost::spirit::x3::lit('"');

static auto const non_double_quote =
    ~boost::spirit::x3::char_('"');

static auto const double_quote_symbol =
    double_quote >> double_quote;

static auto const wkt_latin_text_characters = as<std::string>(
        // Turn skipper off.
        boost::spirit::x3::lexeme[*non_double_quote]);

static auto const quoted_latin_text = as<std::string>(
    double_quote >>
        wkt_latin_text_characters >>
        // // Turn skipper off.
        // boost::spirit::x3::lexeme[*non_double_quote] >>
    double_quote);

// TODO Unicode
static auto const quoted_unicode_text = as<std::string>(
    double_quote >>
        // Turn skipper off.
        boost::spirit::x3::lexeme[*non_double_quote] >>
    double_quote);

static auto const unsigned_integer =
    boost::spirit::x3::uint64;

static auto const signed_integer =
    boost::spirit::x3::int64;

// static auto const unsigned_float =
//     boost::spirit::x3::rule<class unsigned_float, double>() =
//         ~boost::spirit::x3::lit('-') >> boost::spirit::x3::double_
//         ;

// http://boost.2283326.n4.nabble.com/Single-element-attributes-in-X3-quot-still-quot-broken-td4681549.html
// eps should not be necessary from Boost 1.60 onwards:
// http://lists.boost.org/Archives/boost/2016/03/228510.php
static auto const unsigned_float =
    boost::spirit::x3::double_ >> boost::spirit::x3::eps;

static auto const signed_float =
    boost::spirit::x3::double_ >> boost::spirit::x3::eps;

static auto const unsigned_numeric_literal =
    unsigned_float;  // | unsigned_integer

static auto const signed_numeric_literal =
    signed_float;  // | signed_integer

static auto const number =
    signed_numeric_literal | unsigned_numeric_literal;

}  // namespace grammar
}  // namespace wkt
}  // namespace fern
