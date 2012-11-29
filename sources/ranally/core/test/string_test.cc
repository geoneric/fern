#define BOOST_TEST_MODULE ranally core
#include <boost/test/included/unit_test.hpp>
#include "ranally/core/string.h"


BOOST_AUTO_TEST_SUITE(string)

BOOST_AUTO_TEST_CASE(constructor)
{
    ranally::String string;

    string = "aæø";
    BOOST_CHECK_EQUAL(string.encode_in_utf8(), "aæø");
}


BOOST_AUTO_TEST_CASE(strip)
{
    ranally::String string;

    string = ranally::String("bla").strip();
    BOOST_CHECK_EQUAL(string, "bla");

    string = ranally::String("bla").strip("\n");
    BOOST_CHECK_EQUAL(string, "bla");

    string = ranally::String("bla\n").strip();
    BOOST_CHECK_EQUAL(string, "bla");

    string = ranally::String("bla\n").strip("\n");
    BOOST_CHECK_EQUAL(string, "bla");

    string = ranally::String("\nbla").strip();
    BOOST_CHECK_EQUAL(string, "bla");

    string = ranally::String("\nbla").strip("\n");
    BOOST_CHECK_EQUAL(string, "bla");

    string = ranally::String("\nbla\n").strip();
    BOOST_CHECK_EQUAL(string, "bla");

    string = ranally::String("\nbla\n").strip("\n");
    BOOST_CHECK_EQUAL(string, "bla");

    // Strip with argument preserves whitespace.
    string = ranally::String("\n bla \n").strip("\n");
    BOOST_CHECK_EQUAL(string, " bla ");

    // Default strip doesn't preserve whitespace.
    string = ranally::String("\n bla \n").strip();
    BOOST_CHECK_EQUAL(string, "bla");

    string = ranally::String("bla bla").strip("ba");
    BOOST_CHECK_EQUAL(string, "la bl");

    string = ranally::String("").strip();
    BOOST_CHECK_EQUAL(string, "");

    string = ranally::String("").strip("");
    BOOST_CHECK_EQUAL(string, "");

    string = ranally::String("øaø").strip("ø");
    BOOST_CHECK_EQUAL(string, "a");
}

BOOST_AUTO_TEST_SUITE_END()
