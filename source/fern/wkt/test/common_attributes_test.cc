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
    BOOST_CHECK(first == last);
    BOOST_CHECK_EQUAL(value_we_got, value_we_want);
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


BOOST_AUTO_TEST_CASE(length_unit_from_spec)
{
    {
        std::string wkt = R"(LENGTHUNIT["metre",1])";

        fern::wkt::ast::LengthUnit length_unit;
        length_unit.name = "metre";
        length_unit.conversion_factor = 1.0;

        check_parse(wkt, fern::wkt::grammar::length_unit, length_unit);
    }
}
