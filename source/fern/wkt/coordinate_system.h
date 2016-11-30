#pragma once
#include "fern/wkt/ast.h"
#include "fern/wkt/core.h"


namespace fern {
namespace wkt {
namespace grammar {

static auto const cs_type = as<std::string>(
    boost::spirit::x3::lit("affine") |
    boost::spirit::x3::lit("Cartesian") |
    boost::spirit::x3::lit("cylindrical") |
    boost::spirit::x3::lit("ellipsoidal") |
    boost::spirit::x3::lit("linear") |
    boost::spirit::x3::lit("parametric") |
    boost::spirit::x3::lit("polar") |
    boost::spirit::x3::lit("spherical") |
    boost::spirit::x3::lit("temporal") |
    boost::spirit::x3::lit("vertical"));

static auto const dimension = as<uint32_t>(
    unsigned_integer);

axis_name
axis_abbreviation
axis_name_and_abbrev

static auto const axis_name_abbrev = as<ast::AxisNameAbbrev>(
    double_quote >>
        (axis_name | axis_abbreviation | axis_name_and_abbrev) >>
    double_quote);

static auto const axis = as<ast::Axis>(
    boost::spirit::x3::lit("AXIS") >>
    left_delimiter >>
        axis_name_abbrev >>
        wkt_separator >>
        axis_direction >>
        -(wkt_separator >> axis_order) >>
        -(wkt_separator >> axis_unit) >>
        *(wkt_separator >> identifier) >>
    right_delimiter);

static auto const coordinate_system = as<ast::CoordinateSystem>(
    boost::spirit::x3::lit("CS") >>
    left_delimiter >>
        cs_type >>
        wkt_separator >>
        dimension >>
        *(wkt_separator >> identifier) >>
    right_delimiter >>
    +(wkt_separator >> axis)
    -(wkt_separator >> cs_unit));

} // namespace grammar
} // namespace wkt
} // namespace fern
