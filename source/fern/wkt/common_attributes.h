#pragma once
#include <tuple>
#include "fern/wkt/ast.h"
#include "fern/wkt/core.h"


namespace fern {
namespace wkt {
namespace grammar {

// wkt separator: ','

// scope extent identifier remark = 

static auto const scope = as<std::string>(
    // boost::spirit::x3::no_case["SCOPE"] >>
    boost::spirit::x3::lit("SCOPE") >>
    left_delimiter >>
        quoted_latin_text >>
    right_delimiter);

static auto const area = as<std::string>(
    boost::spirit::x3::lit("AREA") >>
    left_delimiter >>
        quoted_latin_text >>
    right_delimiter);

static auto const bbox = as<ast::BBox>(
    boost::spirit::x3::lit("BBOX") >>
    left_delimiter >>
        number >> wkt_separator >>
        number >> wkt_separator >>
        number >> wkt_separator >>
        number >>
    right_delimiter);

static auto const unit_name = as<std::string>(
    quoted_latin_text);

static auto const conversion_factor = as<double>(
    unsigned_numeric_literal);

static auto const length_unit = as<ast::LengthUnit>(
    (boost::spirit::x3::lit("LENGTHUNIT") | boost::spirit::x3::lit("UNIT")) >>
    left_delimiter >>
        unit_name >>
        wkt_separator >>
        conversion_factor >>
        // TODO identifiers
    right_delimiter);

// static auto const vertical_extent =
//     boost::spirit::x3::rule<class vertical_extent,
//             std::vector<double>>() =
//         boost::spirit::x3::lit("VERTICALEXTENT") >>
//         left_delimiter >>
//         (number >> wkt_separator >> number) >>
//         right_delimiter
//         ;

} // namespace grammar
} // namespace wkt
} // namespace fern
