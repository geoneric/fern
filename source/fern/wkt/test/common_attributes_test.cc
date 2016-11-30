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


template<
    typename Parser>
void check_not_parse2(
    std::string const& wkt,
    Parser& parser)
{
    auto first = wkt.begin();
    auto last = wkt.end();
    bool status = x3::phrase_parse(first, last, parser, x3::space);

    BOOST_CHECK(!status);
    BOOST_CHECK(wkt.empty() || first != last);
}


template<
    typename Value,
    typename Parser>
void check_partial_parse(
    std::string const& wkt,
    Parser& parser)
{
    auto first = wkt.begin();
    auto last = wkt.end();
    Value value_we_got;
    bool status = x3::phrase_parse(first, last, parser, x3::space,
        value_we_got);

    BOOST_CHECK(status);
    BOOST_CHECK(first != last);
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


BOOST_AUTO_TEST_CASE(angle_unit_from_spec)
{
    {
        std::string wkt = R"(ANGLEUNIT["degree",0.0174532925199433])";

        fern::wkt::ast::AngleUnit angle_unit;
        angle_unit.name = "degree";
        angle_unit.conversion_factor = 0.0174532925199433;

        check_parse(wkt, fern::wkt::grammar::angle_unit, angle_unit);
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


BOOST_AUTO_TEST_CASE(scale_unit_from_spec)
{
    {
        std::string wkt = R"(SCALEUNIT["parts per million",1E-06])";

        fern::wkt::ast::ScaleUnit scale_unit;
        scale_unit.name = "parts per million";
        scale_unit.conversion_factor = 1E-06;

        check_parse(wkt, fern::wkt::grammar::scale_unit, scale_unit);
    }
}


BOOST_AUTO_TEST_CASE(parametric_unit_from_spec)
{
    {
        std::string wkt = R"(PARAMETRICUNIT["hectopascal",100])";

        fern::wkt::ast::ParametricUnit parametric_unit;
        parametric_unit.name = "hectopascal";
        parametric_unit.conversion_factor = 100.0;

        check_parse(wkt, fern::wkt::grammar::parametric_unit, parametric_unit);
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
    check_not_parse<uint>("123", fern::wkt::grammar::year);
    check_partial_parse<uint>("12345", fern::wkt::grammar::year);
    check_not_parse<uint>("", fern::wkt::grammar::year);

    // TODO [min, max] ?
}


BOOST_AUTO_TEST_CASE(month)
{
    check_parse("12", fern::wkt::grammar::month, 12);
    check_parse("02", fern::wkt::grammar::month, 2);
    check_not_parse<uint>("2", fern::wkt::grammar::month);
    check_partial_parse<uint>("123", fern::wkt::grammar::month);
    check_not_parse<uint>("", fern::wkt::grammar::month);

    // TODO [0, 11], [1, 12] ?
}


BOOST_AUTO_TEST_CASE(day)
{
    check_parse("25", fern::wkt::grammar::day, 25);
    check_parse("02", fern::wkt::grammar::day, 2);
    check_not_parse<uint>("2", fern::wkt::grammar::day);
    check_partial_parse<uint>("123", fern::wkt::grammar::day);
    check_not_parse<uint>("", fern::wkt::grammar::day);

    // TODO [0|1, 365] ?
}


BOOST_AUTO_TEST_CASE(hour)
{
    check_parse("10", fern::wkt::grammar::hour, 10);
    check_parse("02", fern::wkt::grammar::hour, 2);
    check_not_parse<uint>("2", fern::wkt::grammar::hour);
    check_partial_parse<uint>("123", fern::wkt::grammar::hour);
    check_not_parse<uint>("", fern::wkt::grammar::hour);

    // TODO [0|1, 23|24] ?
}


BOOST_AUTO_TEST_CASE(minute)
{
    check_parse("10", fern::wkt::grammar::minute, 10);
    check_parse("02", fern::wkt::grammar::minute, 2);
    check_not_parse<uint>("2", fern::wkt::grammar::minute);
    check_partial_parse<uint>("123", fern::wkt::grammar::minute);
    check_not_parse<uint>("", fern::wkt::grammar::minute);

    // TODO [0|1, 59|60] ?
}


BOOST_AUTO_TEST_CASE(seconds_integer)
{
    check_parse("10", fern::wkt::grammar::seconds_integer, 10);
    check_parse("02", fern::wkt::grammar::seconds_integer, 2);
    check_not_parse<uint>("2", fern::wkt::grammar::seconds_integer);
    check_partial_parse<uint>("123", fern::wkt::grammar::seconds_integer);
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


BOOST_AUTO_TEST_CASE(date_time_from_spec)
{
    {
        std::string wkt = R"(2014)";

        fern::wkt::ast::GregorianCalendarDate calendar_date;
        calendar_date.year = 2014;

        fern::wkt::ast::GregorianCalendarDateTime calendar_date_time;
        calendar_date_time.date = calendar_date;

        fern::wkt::ast::DateTime date_time;
        date_time = calendar_date_time;

        check_parse(wkt, fern::wkt::grammar::date_time, date_time);
    }

    {
        std::string wkt = R"(2014-01)";

        fern::wkt::ast::GregorianCalendarDate calendar_date;
        calendar_date.year = 2014;
        calendar_date.month = 1;

        fern::wkt::ast::GregorianCalendarDateTime calendar_date_time;
        calendar_date_time.date = calendar_date;

        fern::wkt::ast::DateTime date_time;
        date_time = calendar_date_time;

        check_parse(wkt, fern::wkt::grammar::date_time, date_time);
    }

    {
        std::string wkt = R"(2014-03-01)";

        fern::wkt::ast::GregorianCalendarDate calendar_date;
        calendar_date.year = 2014;
        calendar_date.month = 3;
        calendar_date.day = 1;

        fern::wkt::ast::GregorianCalendarDateTime calendar_date_time;
        calendar_date_time.date = calendar_date;

        fern::wkt::ast::DateTime date_time;
        date_time = calendar_date_time;

        check_parse(wkt, fern::wkt::grammar::date_time, date_time);
    }

    // TODO String is not parsed as ordinal date. Probably because
    //     month parser is too general. It should not parse 060 as a month.
    //     Ordinal day parser must always parse 3 digits.
    {
        std::string wkt = R"(2014-060)";

        fern::wkt::ast::GregorianOrdinalDate ordinal_date;
        ordinal_date.year = 2014;
        ordinal_date.day = 60;

        fern::wkt::ast::GregorianOrdinalDateTime ordinal_date_time;
        ordinal_date_time.date = ordinal_date;

        check_parse(wkt, fern::wkt::grammar::gregorian_ordinal_date_time,
            ordinal_date_time);
        // TODO Both parsers parse the same string, while only the
        //      ordinal one should...
        //      The calendar parser parses the string partly.
        // check_not_parse2(wkt,
        //     fern::wkt::grammar::gregorian_calendar_date_time);

        fern::wkt::ast::DateTime date_time;
        date_time = ordinal_date_time;

        // TODO
        // check_parse(wkt, fern::wkt::grammar::date_time, date_time);
    }

    {
        std::string wkt = R"(2014-05-06T23Z)";

        fern::wkt::ast::GregorianCalendarDate calendar_date;
        calendar_date.year = 2014;
        calendar_date.month = 5;
        calendar_date.day = 6;

        fern::wkt::ast::TimeZoneDesignator time_zone;
        time_zone = 'Z';

        fern::wkt::ast::HourClock hour_clock;
        hour_clock.hour = 23;
        hour_clock.time_zone = time_zone;

        fern::wkt::ast::GregorianCalendarDateTime calendar_date_time;
        calendar_date_time.date = calendar_date;
        calendar_date_time.time = hour_clock;

        fern::wkt::ast::DateTime date_time;
        date_time = calendar_date_time;

        check_parse(wkt, fern::wkt::grammar::date_time, date_time);
    }

    {
        std::string wkt = R"(2014-157T23Z)";

        fern::wkt::ast::GregorianOrdinalDate ordinal_date;
        ordinal_date.year = 2014;
        ordinal_date.day = 157;

        fern::wkt::ast::TimeZoneDesignator time_zone;
        time_zone = 'Z';

        fern::wkt::ast::HourClock hour_clock;
        hour_clock.hour = 23;
        hour_clock.time_zone = time_zone;

        fern::wkt::ast::GregorianOrdinalDateTime ordinal_date_time;
        ordinal_date_time.date = ordinal_date;
        ordinal_date_time.time = hour_clock;

        fern::wkt::ast::DateTime date_time;
        date_time = ordinal_date_time;

        // TODO See above.
        // check_parse(wkt, fern::wkt::grammar::date_time, date_time);
    }

    {
        std::string wkt = R"(2014-07-12T16:00Z)";

        fern::wkt::ast::GregorianCalendarDate calendar_date;
        calendar_date.year = 2014;
        calendar_date.month = 7;
        calendar_date.day = 12;

        fern::wkt::ast::TimeZoneDesignator time_zone;
        time_zone = 'Z';

        fern::wkt::ast::HourClock hour_clock;
        hour_clock.hour = 16;
        hour_clock.minute = 0;
        hour_clock.time_zone = time_zone;

        fern::wkt::ast::GregorianCalendarDateTime calendar_date_time;
        calendar_date_time.date = calendar_date;
        calendar_date_time.time = hour_clock;

        fern::wkt::ast::DateTime date_time;
        date_time = calendar_date_time;

        check_parse(wkt, fern::wkt::grammar::date_time, date_time);
    }

    {
        std::string wkt = R"(2014-07-12T17:00+01)";

        fern::wkt::ast::GregorianCalendarDate calendar_date;
        calendar_date.year = 2014;
        calendar_date.month = 7;
        calendar_date.day = 12;

        fern::wkt::ast::LocalTimeZoneDesignator local_time_zone;
        local_time_zone.sign = '+';
        local_time_zone.hour = 1;

        fern::wkt::ast::TimeZoneDesignator time_zone;
        time_zone = local_time_zone;

        fern::wkt::ast::HourClock hour_clock;
        hour_clock.hour = 17;
        hour_clock.minute = 0;
        hour_clock.time_zone = time_zone;

        fern::wkt::ast::GregorianCalendarDateTime calendar_date_time;
        calendar_date_time.date = calendar_date;
        calendar_date_time.time = hour_clock;

        fern::wkt::ast::DateTime date_time;
        date_time = calendar_date_time;

        check_parse(wkt, fern::wkt::grammar::date_time, date_time);
    }

    {
        std::string wkt = R"(2014-09-18T08:17:56Z)";

        fern::wkt::ast::GregorianCalendarDate calendar_date;
        calendar_date.year = 2014;
        calendar_date.month = 9;
        calendar_date.day = 18;

        fern::wkt::ast::TimeZoneDesignator time_zone;
        time_zone = 'Z';

        fern::wkt::ast::Second second;
        second.integer = 56;

        fern::wkt::ast::HourClock hour_clock;
        hour_clock.hour = 8;
        hour_clock.minute = 17;
        hour_clock.second = second;
        hour_clock.time_zone = time_zone;

        fern::wkt::ast::GregorianCalendarDateTime calendar_date_time;
        calendar_date_time.date = calendar_date;
        calendar_date_time.time = hour_clock;

        fern::wkt::ast::DateTime date_time;
        date_time = calendar_date_time;

        check_parse(wkt, fern::wkt::grammar::date_time, date_time);
    }

    {
        std::string wkt = R"(2014-11-23T00:34:56.789Z)";

        fern::wkt::ast::GregorianCalendarDate calendar_date;
        calendar_date.year = 2014;
        calendar_date.month = 11;
        calendar_date.day = 23;

        fern::wkt::ast::TimeZoneDesignator time_zone;
        time_zone = 'Z';

        fern::wkt::ast::Second second;
        second.integer = 56;
        second.fraction = 789;

        fern::wkt::ast::HourClock hour_clock;
        hour_clock.hour = 0;
        hour_clock.minute = 34;
        hour_clock.second = second;
        hour_clock.time_zone = time_zone;

        fern::wkt::ast::GregorianCalendarDateTime calendar_date_time;
        calendar_date_time.date = calendar_date;
        calendar_date_time.time = hour_clock;

        fern::wkt::ast::DateTime date_time;
        date_time = calendar_date_time;

        check_parse(wkt, fern::wkt::grammar::date_time, date_time);
    }
}


BOOST_AUTO_TEST_CASE(temporal_extent_from_spec)
{
    {
        std::string wkt =
            R"(TIMEEXTENT[2013-01-01,2013-12-31])";

        fern::wkt::ast::GregorianCalendarDate start_date, end_date;
        start_date.year = 2013;
        start_date.month = 1;
        start_date.day = 1;
        end_date.year = 2013;
        end_date.month = 12;
        end_date.day = 31;

        fern::wkt::ast::GregorianCalendarDateTime start, end;
        start.date = start_date;
        end.date = end_date;

        fern::wkt::ast::TemporalExtent temporal_extent;
        temporal_extent.start = start;
        temporal_extent.end = end;

        check_parse(wkt, fern::wkt::grammar::temporal_extent,
            temporal_extent);
    }

    {
        std::string wkt =
            R"(TIMEEXTENT["Jurassic","Quaternary"])";

        fern::wkt::ast::TemporalExtent temporal_extent;
        temporal_extent.start = std::string("Jurassic");
        temporal_extent.end = std::string("Quaternary");

        check_parse(wkt, fern::wkt::grammar::temporal_extent,
            temporal_extent);
    }
}


BOOST_AUTO_TEST_CASE(remark_from_spec)
{
    {
        std::string wkt =
            R"(REMARK["A remark in ASCII"])";
        std::string remark{"A remark in ASCII"};

        check_parse(wkt, fern::wkt::grammar::remark, remark);
    }

    {
        std::string wkt =
            R"(REMARK["Замечание на русском языке"])";
        std::string remark{"Замечание на русском языке"};
    }

    // Замечание на русском языке
}
