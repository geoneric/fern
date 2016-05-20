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


namespace x3 = boost::spirit::x3;


BOOST_AUTO_TEST_CASE(scope_example_from_spec)
{
    std::string wkt =
        R"(SCOPE["Large scale topographic mapping and cadastre."])";
    std::string scope;
    auto first = wkt.begin();
    auto last = wkt.end();
    bool status = x3::phrase_parse(first, last, fern::wkt::scope,
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
    bool status = x3::phrase_parse(first, last, fern::wkt::area,
        x3::space, area);

    BOOST_CHECK(status);
    BOOST_CHECK(first == last);
    BOOST_CHECK_EQUAL(area, "Netherlands offshore.");
}


BOOST_AUTO_TEST_CASE(bbox_example_from_spec)
{
    std::string wkt =
        R"(BBOX[51.43,2.54,55.77,6.40])";
    std::vector<double> bbox;
    auto first = wkt.begin();
    auto last = wkt.end();
    bool status = x3::phrase_parse(first, last, fern::wkt::bbox,
        x3::space, bbox);

    BOOST_CHECK(status);
    BOOST_CHECK(first == last);
    BOOST_REQUIRE_EQUAL(bbox.size(), 4);
    BOOST_CHECK_EQUAL(bbox[0], 51.43);
    BOOST_CHECK_EQUAL(bbox[1], 2.54);
    BOOST_CHECK_EQUAL(bbox[2], 55.77);
    BOOST_CHECK_EQUAL(bbox[3], 6.40);
}


BOOST_AUTO_TEST_CASE(unsigned_integer)
{
    {
        std::string wkt = R"(51)";
        uint64_t unsigned_integer;
        auto first = wkt.begin();
        auto last = wkt.end();
        bool status = x3::phrase_parse(first, last,
            fern::wkt::unsigned_integer, x3::space, unsigned_integer);

        BOOST_CHECK(status);
        BOOST_CHECK(first == last);
        BOOST_CHECK_EQUAL(unsigned_integer, 51);
    }

    {
        std::string wkt = R"(-51)";
        uint64_t unsigned_integer;
        auto first = wkt.begin();
        auto last = wkt.end();
        bool status = x3::phrase_parse(first, last,
            fern::wkt::unsigned_integer, x3::space, unsigned_integer);

        BOOST_CHECK(!status);
        BOOST_CHECK(first != last);
    }
}


BOOST_AUTO_TEST_CASE(signed_integer)
{
    {
        std::string wkt = R"(51)";
        uint64_t signed_integer;
        auto first = wkt.begin();
        auto last = wkt.end();
        bool status = x3::phrase_parse(first, last,
            fern::wkt::signed_integer, x3::space, signed_integer);

        BOOST_CHECK(status);
        BOOST_CHECK(first == last);
        BOOST_CHECK_EQUAL(signed_integer, 51);
    }

    // TODO Add support for negative sign.
    // {
    //     std::string wkt = R"(-51)";
    //     uint64_t signed_integer;
    //     auto first = wkt.begin();
    //     auto last = wkt.end();
    //     bool status = x3::phrase_parse(first, last,
    //         fern::wkt::signed_integer, x3::space, signed_integer);

    //     BOOST_CHECK(status);
    //     BOOST_CHECK(first != last);
    //     BOOST_CHECK_EQUAL(signed_integer, -51);
    // }
}


BOOST_AUTO_TEST_CASE(unsigned_float)
{
    {
        std::string wkt = R"(51)";
        double unsigned_float;
        auto first = wkt.begin();
        auto last = wkt.end();
        bool status = x3::phrase_parse(first, last,
            fern::wkt::unsigned_float, x3::space, unsigned_float);

        BOOST_CHECK(status);
        BOOST_CHECK(first == last);
        BOOST_CHECK_EQUAL(unsigned_float, 51);
    }

    {
        std::string wkt = R"(-51)";
        double unsigned_float;
        auto first = wkt.begin();
        auto last = wkt.end();
        /* bool status = */ x3::phrase_parse(first, last,
            fern::wkt::unsigned_float, x3::space, unsigned_float);

        // TODO Update rule to not accept unary minus.
        // BOOST_CHECK(!status);
        // BOOST_CHECK(first != last);
    }
}


BOOST_AUTO_TEST_CASE(signed_float)
{
    {
        std::string wkt = R"(51.3)";
        double signed_float;
        auto first = wkt.begin();
        auto last = wkt.end();
        bool status = x3::phrase_parse(first, last,
            fern::wkt::signed_float, x3::space, signed_float);

        BOOST_CHECK(status);
        BOOST_CHECK(first == last);
        BOOST_CHECK_EQUAL(signed_float, 51.3);
    }

    {
        std::string wkt = R"(-51.3)";
        double signed_float;
        auto first = wkt.begin();
        auto last = wkt.end();
        bool status = x3::phrase_parse(first, last,
            fern::wkt::signed_float, x3::space, signed_float);

        BOOST_CHECK(status);
        BOOST_CHECK(first == last);
        BOOST_CHECK_EQUAL(signed_float, -51.3);
    }
}


BOOST_AUTO_TEST_CASE(conversion_factor)
{
    {
        std::string wkt = R"(1)";
        double conversion_factor;
        auto first = wkt.begin();
        auto last = wkt.end();
        bool status = x3::phrase_parse(first, last,
            fern::wkt::conversion_factor, x3::space, conversion_factor);

        BOOST_CHECK(status);
        BOOST_CHECK(first == last);
        BOOST_CHECK_EQUAL(conversion_factor, 1);
    }

    {
        std::string wkt = R"(1.0000135965)";
        double conversion_factor;
        auto first = wkt.begin();
        auto last = wkt.end();
        bool status = x3::phrase_parse(first, last,
            fern::wkt::conversion_factor, x3::space, conversion_factor);

        BOOST_CHECK(status);
        BOOST_CHECK(first == last);
        BOOST_CHECK_EQUAL(conversion_factor, 1.0000135965);
    }
}


BOOST_AUTO_TEST_CASE(length_unit_from_spec)
{
    std::string wkt =
        R"(LENGTHUNIT["metre",1])";

    std::tuple<std::string, double> length_unit;
    auto first = wkt.begin();
    auto last = wkt.end();

    bool status = x3::phrase_parse(first, last, fern::wkt::length_unit,
        x3::space, length_unit);

    BOOST_CHECK(status);
    BOOST_CHECK(first == last);
    BOOST_CHECK_EQUAL(std::get<0>(length_unit), "metre");
    BOOST_CHECK_EQUAL(std::get<1>(length_unit), 0.0);
}


