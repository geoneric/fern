// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern wkt common_attributes
#include <boost/test/unit_test.hpp>
#include "fern/wkt/common_attributes.h"


// http://stackoverflow.com/questions/34559607/attributes-from-boost-spirit-grammar-error-from-stdvector-of-boostvariant
// http://stackoverflow.com/questions/34566179/overloaded-output-operator-not-found-for-boost-spirit-expression


namespace x3 = boost::spirit::x3;


template<
    typename Parser,
    typename Value>
void check_parse(
    std::string const& wkt,
    Parser& parser,
    Value const& value_we_want)
{
    auto first = wkt.begin();
    auto last = wkt.end();
    Value value_we_got;
    bool status = x3::phrase_parse(first, last, parser, x3::space,
        value_we_got);

    BOOST_CHECK(status);

    if(status) {
        BOOST_CHECK(first == last);
        BOOST_CHECK_EQUAL(value_we_got, value_we_want);
    }
}


template<
    typename Value,
    typename Parser>
void check_not_parse(
    std::string const& wkt,
    Parser& parser)
{
    auto first = wkt.begin();
    auto last = wkt.end();
    Value value_we_got;
    bool status = x3::phrase_parse(first, last, parser, x3::space,
        value_we_got);

    BOOST_CHECK(!status);
    BOOST_CHECK(wkt.empty() || first != last);
}


BOOST_AUTO_TEST_CASE(scope_example_from_spec)
{
    std::string wkt =
        R"(SCOPE["Large scale topographic mapping and cadastre."])";
    std::string scope;
    auto first = wkt.begin();
    auto last = wkt.end();
    bool status = x3::phrase_parse(first, last, fern::wkt::grammar::scope,
        x3::space, scope);

    BOOST_CHECK(status);
    BOOST_CHECK(first == last);
    BOOST_CHECK_EQUAL(scope, "Large scale topographic mapping and cadastre.");
}


BOOST_AUTO_TEST_CASE(area_example_from_spec)
{
    std::string wkt =
        R"(AREA["Netherlands offshore."])";
    std::string area;
    auto first = wkt.begin();
    auto last = wkt.end();
    bool status = x3::phrase_parse(first, last, fern::wkt::grammar::area,
        x3::space, area);

    BOOST_CHECK(status);
    BOOST_CHECK(first == last);
    BOOST_CHECK_EQUAL(area, "Netherlands offshore.");
}


BOOST_AUTO_TEST_CASE(bbox_example_from_spec)
{
    std::string wkt = R"(BBOX[51.43,2.54,55.77,6.40])";

    fern::wkt::ast::BBox bbox;
    bbox.lower_left_latitude = 51.43;
    bbox.lower_left_longitude = 2.54;
    bbox.upper_right_latitude = 55.77;
    bbox.upper_right_longitude = 6.40;

    check_parse(wkt, fern::wkt::grammar::bbox, bbox);
}


BOOST_AUTO_TEST_CASE(unsigned_integer)
{
    {
        std::string wkt = R"(51)";
        uint64_t unsigned_integer{51};
        check_parse(wkt, fern::wkt::grammar::unsigned_integer,
            unsigned_integer);
    }

    {
        std::string wkt = R"(-51)";
        uint64_t unsigned_integer;
        auto first = wkt.begin();
        auto last = wkt.end();
        bool status = x3::phrase_parse(first, last,
            fern::wkt::grammar::unsigned_integer, x3::space, unsigned_integer);

        BOOST_CHECK(!status);
        BOOST_CHECK(first != last);
    }
}


BOOST_AUTO_TEST_CASE(signed_integer)
{
    {
        std::string wkt = R"(51)";
        uint64_t signed_float{51};
        check_parse(wkt, fern::wkt::grammar::signed_float, signed_float);
    }

    // TODO Add support for negative sign.
    // {
    //     std::string wkt = R"(-51)";
    //     uint64_t signed_integer;
    //     auto first = wkt.begin();
    //     auto last = wkt.end();
    //     bool status = x3::phrase_parse(first, last,
    //         fern::wkt::grammar::signed_integer, x3::space, signed_integer);

    //     BOOST_CHECK(status);
    //     BOOST_CHECK(first != last);
    //     BOOST_CHECK_EQUAL(signed_integer, -51);
    // }
}


BOOST_AUTO_TEST_CASE(unsigned_float)
{
    {
        std::string wkt = R"(51)";
        double unsigned_float{51.0};
        check_parse(wkt, fern::wkt::grammar::unsigned_float, unsigned_float);
    }

    {
        std::string wkt = R"(-51)";
        double unsigned_float;
        auto first = wkt.begin();
        auto last = wkt.end();
        /* bool status = */ x3::phrase_parse(first, last,
            fern::wkt::grammar::unsigned_float, x3::space, unsigned_float);

        // TODO Update rule to not accept unary minus.
        // BOOST_CHECK(!status);
        // BOOST_CHECK(first != last);
    }
}


BOOST_AUTO_TEST_CASE(signed_float)
{
    {
        std::string wkt = R"(51.3)";
        double signed_float{51.3};
        check_parse(wkt, fern::wkt::grammar::signed_float, signed_float);
    }

    {
        std::string wkt = R"(-51.3)";
        double signed_float{-51.3};
        check_parse(wkt, fern::wkt::grammar::signed_float, signed_float);
    }
}


BOOST_AUTO_TEST_CASE(conversion_factor)
{
    {
        std::string wkt = R"(1)";
        double conversion_factor{1.0};
        check_parse(wkt, fern::wkt::grammar::conversion_factor,
            conversion_factor);
    }

    {
        std::string wkt = R"(1.0000135965)";
        double conversion_factor{1.0000135965};
        check_parse(wkt, fern::wkt::grammar::conversion_factor,
            conversion_factor);
    }
}


BOOST_AUTO_TEST_CASE(authority_unique_identifier)
{
    {
        std::string wkt = R"("Abcd_Ef")";

        fern::wkt::ast::AuthorityUniqueIdentifier identifier;
        identifier = "Abcd_Ef";

        check_parse(wkt, fern::wkt::grammar::authority_unique_identifier,
            identifier);
    }

    {
        std::string wkt = R"(4326)";

        fern::wkt::ast::AuthorityUniqueIdentifier identifier;
        identifier = 4326;

        check_parse(wkt, fern::wkt::grammar::authority_unique_identifier,
            identifier);
    }
}


BOOST_AUTO_TEST_CASE(version)
{
    {
        std::string wkt = R"(7.1)";

        fern::wkt::ast::Version version;
        version = 7.1;

        check_parse(wkt, fern::wkt::grammar::version, version);
    }

    {
        std::string wkt = R"("2001-04-20")";

        fern::wkt::ast::Version version;
        version = "2001-04-20";

        check_parse(wkt, fern::wkt::grammar::version, version);
    }
}


BOOST_AUTO_TEST_CASE(identifier_from_spec)
{
    {
        std::string wkt = R"(ID["Authority name","Abcd_Ef",7.1])";

        fern::wkt::ast::Identifier identifier;
        identifier.authority_name = "Authority name";
        identifier.authority_unique_identifier = "Abcd_Ef";
        identifier.version = 7.1;

        check_parse(wkt, fern::wkt::grammar::identifier, identifier);
    }

    {
        std::string wkt = R"(ID["EPSG",4326])";

        fern::wkt::ast::Identifier identifier;
        identifier.authority_name = "EPSG";
        identifier.authority_unique_identifier = 4326;

        check_parse(wkt, fern::wkt::grammar::identifier, identifier);
    }

    {
        std::string wkt =
            R"(ID["EPSG",4326,URI["urn:ogc:def:crs:EPSG::4326"]])";

        fern::wkt::ast::Identifier identifier;
        identifier.authority_name = "EPSG";
        identifier.authority_unique_identifier = 4326;
        identifier.id_uri = "urn:ogc:def:crs:EPSG::4326";

        check_parse(wkt, fern::wkt::grammar::identifier, identifier);
    }

    {
        std::string wkt =
            R"(ID["EuroGeographics","ES_ED50(BAL99) to ETRS89","2001-04-20"])";

        fern::wkt::ast::Identifier identifier;
        identifier.authority_name = "EuroGeographics";
        identifier.authority_unique_identifier = "ES_ED50(BAL99) to ETRS89";
        identifier.version = "2001-04-20";

        check_parse(wkt, fern::wkt::grammar::identifier, identifier);
    }
}


BOOST_AUTO_TEST_CASE(length_unit_from_spec)
{
    {
        std::string wkt = R"(LENGTHUNIT["metre",1])";

        fern::wkt::ast::LengthUnit length_unit;
        length_unit.name = "metre";
        length_unit.conversion_factor = 1.0;

        check_parse(wkt, fern::wkt::grammar::length_unit, length_unit);
    }

    {
        std::string wkt = R"(LENGTHUNIT["German legal metre",1.0000135965])";

        fern::wkt::ast::LengthUnit length_unit;
        length_unit.name = "German legal metre";
        length_unit.conversion_factor = 1.0000135965;

        check_parse(wkt, fern::wkt::grammar::length_unit, length_unit);
    }
}


BOOST_AUTO_TEST_CASE(vertical_extent_from_spec)
{
    {
        std::string wkt =
            R"(VERTICALEXTENT[-1000,0,LENGTHUNIT["metre",1.0]])";

        fern::wkt::ast::VerticalExtent vertical_extent;
        vertical_extent.minimum_height = -1000.0;
        vertical_extent.maximum_height = 0.0;
        vertical_extent.length_unit = fern::wkt::ast::LengthUnit{};
        vertical_extent.length_unit->name = "metre";
        vertical_extent.length_unit->conversion_factor = 1.0;

        check_parse(wkt, fern::wkt::grammar::vertical_extent,
            vertical_extent);
    }

    {
        std::string wkt =
            R"(VERTICALEXTENT[-1000,0])";

        fern::wkt::ast::VerticalExtent vertical_extent;
        vertical_extent.minimum_height = -1000.0;
        vertical_extent.maximum_height = 0.0;

        check_parse(wkt, fern::wkt::grammar::vertical_extent,
            vertical_extent);
    }
}


BOOST_AUTO_TEST_CASE(year)
{
    check_parse("1234", fern::wkt::grammar::year, 1234);
    // TODO
    // check_not_parse<uint>("123", fern::wkt::grammar::year);
    // check_not_parse<uint>("12345", fern::wkt::grammar::year);
    check_not_parse<uint>("", fern::wkt::grammar::year);

    // TODO [min, max] ?
}


BOOST_AUTO_TEST_CASE(month)
{
    check_parse("12", fern::wkt::grammar::month, 12);
    check_parse("02", fern::wkt::grammar::month, 2);
    // TODO
    // check_not_parse<uint>("2", fern::wkt::grammar::month);
    check_not_parse<uint>("", fern::wkt::grammar::month);

    // TODO [0, 11], [1, 12] ?
}


BOOST_AUTO_TEST_CASE(day)
{
    check_parse("25", fern::wkt::grammar::day, 25);
    check_parse("02", fern::wkt::grammar::day, 2);
    // TODO
    // check_not_parse<uint>("2", fern::wkt::grammar::day);
    check_not_parse<uint>("", fern::wkt::grammar::day);

    // TODO [0|1, 365] ?
}


BOOST_AUTO_TEST_CASE(hour)
{
    check_parse("10", fern::wkt::grammar::hour, 10);
    check_parse("02", fern::wkt::grammar::hour, 2);
    // TODO
    // check_not_parse<uint>("2", fern::wkt::grammar::hour);
    check_not_parse<uint>("", fern::wkt::grammar::hour);

    // TODO [0|1, 23|24] ?
}


BOOST_AUTO_TEST_CASE(minute)
{
    check_parse("10", fern::wkt::grammar::minute, 10);
    check_parse("02", fern::wkt::grammar::minute, 2);
    // TODO
    // check_not_parse<uint>("2", fern::wkt::grammar::minute);
    check_not_parse<uint>("", fern::wkt::grammar::minute);

    // TODO [0|1, 59|60] ?
}


BOOST_AUTO_TEST_CASE(seconds_integer)
{
    check_parse("10", fern::wkt::grammar::seconds_integer, 10);
    check_parse("2", fern::wkt::grammar::seconds_integer, 2);
    check_not_parse<uint>("", fern::wkt::grammar::seconds_integer);

    // TODO [0|1, 59|60] ?
}


BOOST_AUTO_TEST_CASE(seconds_fraction)
{
    check_parse("10", fern::wkt::grammar::seconds_fraction, 10);
    check_parse("2", fern::wkt::grammar::seconds_fraction, 2);
    check_not_parse<uint>("", fern::wkt::grammar::seconds_fraction);
}


BOOST_AUTO_TEST_CASE(second)
{
    {
        std::string wkt = R"(10)";

        fern::wkt::ast::Second second;
        second.integer = 10;

        check_parse(wkt, fern::wkt::grammar::second, second);
    }

    {
        std::string wkt = R"(10.5)";

        fern::wkt::ast::Second second;
        second.integer = 10;
        second.fraction = 5;

        check_parse(wkt, fern::wkt::grammar::second, second);
    }
}


BOOST_AUTO_TEST_CASE(local_time_zone_designator)
{
    {
        std::string wkt = R"(+03)";

        fern::wkt::ast::LocalTimeZoneDesignator designator;
        designator.sign = '+';
        designator.hour = 3;

        check_parse(wkt, fern::wkt::grammar::local_time_zone_designator,
            designator);
    }

    {
        std::string wkt = R"(-03:45)";

        fern::wkt::ast::LocalTimeZoneDesignator designator;
        designator.sign = '-';
        designator.hour = 3;
        designator.minute = 45;

        check_parse(wkt, fern::wkt::grammar::local_time_zone_designator,
            designator);
    }
}


BOOST_AUTO_TEST_CASE(time_zone_designator)
{
    {
        std::string wkt = R"(+03)";

        fern::wkt::ast::LocalTimeZoneDesignator local_designator;
        local_designator.sign = '+';
        local_designator.hour = 3;

        fern::wkt::ast::TimeZoneDesignator designator = local_designator;

        check_parse(wkt, fern::wkt::grammar::time_zone_designator,
            designator);
    }

    {
        std::string wkt = R"(Z)";
        fern::wkt::ast::TimeZoneDesignator designator = 'Z';
        check_parse(wkt, fern::wkt::grammar::time_zone_designator,
            designator);
    }
}


BOOST_AUTO_TEST_CASE(gregorian_calendar_date)
{
    {
        std::string wkt = R"(2016)";

        fern::wkt::ast::GregorianCalendarDate date;
        date.year = 2016;

        check_parse(wkt, fern::wkt::grammar::gregorian_calendar_date, date);
    }

    {
        std::string wkt = R"(2016-05)";

        fern::wkt::ast::GregorianCalendarDate date;
        date.year = 2016;
        date.month = 5;

        check_parse(wkt, fern::wkt::grammar::gregorian_calendar_date, date);
    }

    {
        std::string wkt = R"(2016-05-25)";

        fern::wkt::ast::GregorianCalendarDate date;
        date.year = 2016;
        date.month = 5;
        date.day = 25;

        check_parse(wkt, fern::wkt::grammar::gregorian_calendar_date, date);
    }
}


BOOST_AUTO_TEST_CASE(hour_clock)
{
    {
        std::string wkt = R"(T08+01)";

        fern::wkt::ast::LocalTimeZoneDesignator local_designator;
        local_designator.sign = '+';
        local_designator.hour = 1;

        fern::wkt::ast::HourClock clock;
        clock.hour = 8;
        clock.time_zone = local_designator;

        check_parse(wkt, fern::wkt::grammar::hour_clock, clock);
    }

    {
        std::string wkt = R"(T08:04:33Z)";

        fern::wkt::ast::Second second;
        second.integer = 33;

        fern::wkt::ast::HourClock clock;
        clock.hour = 8;
        clock.minute = 4;
        clock.second = second;
        clock.time_zone = 'Z';

        check_parse(wkt, fern::wkt::grammar::hour_clock, clock);
    }
}


// BOOST_AUTO_TEST_CASE(temporal_extent_from_spec)
// {
//     {
//         std::string wkt =
//             R"(TIMEEXTENT[2013-01-01,2013-12-31])";
// 
//         fern::wkt::ast::TemporalExtent temporal_extent;
//         temporal_extent.start = xxx;
//         temporal_extent.end = xxx;
// 
//         check_parse(wkt, fern::wkt::grammar::temporal_extent,
//             temporal_extent);
//     }
// 
//     {
//         std::string wkt =
//             R"(TIMEEXTENT["Jurassic","Quaternary"])";
// 
//         fern::wkt::ast::TemporalExtent temporal_extent;
//         temporal_extent.start = "Jurassic";
//         temporal_extent.end = "Quaternary";
// 
//         check_parse(wkt, fern::wkt::grammar::temporal_extent,
//             temporal_extent);
//     }
// }


