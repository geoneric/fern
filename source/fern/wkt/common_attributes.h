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

static auto const authority_name = as<std::string>(
    quoted_latin_text);

static auto const authority_unique_identifier =
    as<ast::AuthorityUniqueIdentifier>(
        number | quoted_latin_text);

static auto const version =
    as<ast::Version>(
        number | quoted_latin_text);

static auto const citation = as<std::string>(
    quoted_latin_text);

static auto const authority_citation = as<std::string>(
    boost::spirit::x3::lit("CITATION") >>
    left_delimiter >>
        citation >>
    right_delimiter);

static auto const uri = as<std::string>(
    quoted_latin_text);

static auto const id_uri = as<std::string>(
    boost::spirit::x3::lit("URI") >>
    left_delimiter >>
        uri >>
    right_delimiter);

static auto const identifier = as<ast::Identifier>(
    boost::spirit::x3::lit("ID") >>
    left_delimiter >>
        authority_name >>
        wkt_separator >>
        authority_unique_identifier >>
        -(wkt_separator >> version) >>
        -(wkt_separator >> authority_citation) >>
        -(wkt_separator >> id_uri) >>
    right_delimiter);

static auto const conversion_factor = as<double>(
    unsigned_numeric_literal);

static auto const unit_name = as<std::string>(
    quoted_latin_text);

static auto const length_unit = as<ast::LengthUnit>(
    // UNIT is deprecated.
    (boost::spirit::x3::lit("LENGTHUNIT") | boost::spirit::x3::lit("UNIT")) >>
    left_delimiter >>
        unit_name >>
        wkt_separator >>
        conversion_factor >>
        *(wkt_separator >> identifier) >>
    right_delimiter);

static auto const vertical_extent = as<ast::VerticalExtent>(
    boost::spirit::x3::lit("VERTICALEXTENT") >>
    left_delimiter >>
        number >> wkt_separator >> number >>
        -(wkt_separator >> length_unit) >>
    right_delimiter);


// TODO Four digits.
static auto const year = unsigned_integer;

// TODO Two digits, leading zero if less than 10.
static auto const month = unsigned_integer;

// TODO Two digits, leading zero if less than 10.
static auto const day = unsigned_integer;

// TODO Two digits, leading zero if less than 10.
static auto const hour = unsigned_integer;

// TODO Two digits, leading zero if less than 10.
static auto const minute = unsigned_integer;

// TODO Two digits, leading zero if less than 10.
static auto const seconds_integer = unsigned_integer;

// TODO Two digits, leading zero if less than 10.
static auto const seconds_fraction = unsigned_integer;

static auto const second = as<ast::Second>(
    seconds_integer >> -(period >> -seconds_fraction));

static auto const utc_designator =
    boost::spirit::x3::char_('Z');

static auto const local_time_zone_designator =
    as<ast::LocalTimeZoneDesignator>(
        (boost::spirit::x3::char_('-') | boost::spirit::x3::char_('+')) >>
        hour >>
       -(colon >> minute));

static auto const time_zone_designator = as<ast::TimeZoneDesignator>(
    utc_designator | local_time_zone_designator);

static auto const time_designator =
    boost::spirit::x3::lit('T');

static auto const gregorian_calendar_date = as<ast::GregorianCalendarDate>(
    year >> -(hyphen >> month >> -(hyphen >> day)));

static auto const hour_clock = as<ast::HourClock>(
    time_designator >>
    hour >>
    -(colon >> minute >> -(colon >> second)) >>
    time_zone_designator);

static auto const gregorian_calendar_date_time =
    as<ast::GregorianCalendarDateTime>(
        gregorian_calendar_date >>
        -hour_clock);

// TODO Three digits, leading zeros if less than 100.
static auto const ordinal_day = unsigned_integer;

static auto const gregorian_ordinal_date =
    as<ast::GregorianOrdinalDate>(
        year >> -(hyphen >> ordinal_day));

static auto const gregorian_ordinal_date_time =
    as<ast::GregorianOrdinalDateTime>(
        gregorian_ordinal_date >>
        -hour_clock);

// static auto const date_time = as<ast::DateTime>(
//     gregorian_calendar_date_time | gregorian_ordinal_date_time);
// 
// static auto const temporal_extent = as<ast::TemporalExtent>(
//     boost::spirit::x3::lit("TIMEEXTENT") >>
//     left_delimiter >>
//         -(date_time | quoted_latin_text) >>
//         -(date_time | quoted_latin_text) >>
//     right_delimiter);






} // namespace grammar
} // namespace wkt
} // namespace fern
