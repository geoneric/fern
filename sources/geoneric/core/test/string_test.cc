#define BOOST_TEST_MODULE geoneric core
#include <boost/test/unit_test.hpp>
#include "geoneric/core/string.h"


BOOST_AUTO_TEST_SUITE(string)

BOOST_AUTO_TEST_CASE(constructor)
{
    geoneric::String string;

    string = "aæø";
    BOOST_CHECK_EQUAL(string.encode_in_utf8(), "aæø");
}


BOOST_AUTO_TEST_CASE(strip)
{
    geoneric::String string;

    string = geoneric::String("bla").strip();
    BOOST_CHECK_EQUAL(string, "bla");

    string = geoneric::String("bla").strip("\n");
    BOOST_CHECK_EQUAL(string, "bla");

    string = geoneric::String("bla\n").strip();
    BOOST_CHECK_EQUAL(string, "bla");

    string = geoneric::String("bla\n").strip("\n");
    BOOST_CHECK_EQUAL(string, "bla");

    string = geoneric::String("\nbla").strip();
    BOOST_CHECK_EQUAL(string, "bla");

    string = geoneric::String("\nbla").strip("\n");
    BOOST_CHECK_EQUAL(string, "bla");

    string = geoneric::String("\nbla\n").strip();
    BOOST_CHECK_EQUAL(string, "bla");

    string = geoneric::String("\nbla\n").strip("\n");
    BOOST_CHECK_EQUAL(string, "bla");

    // Strip with argument preserves whitespace.
    string = geoneric::String("\n bla \n").strip("\n");
    BOOST_CHECK_EQUAL(string, " bla ");

    // Default strip doesn't preserve whitespace.
    string = geoneric::String("\n bla \n").strip();
    BOOST_CHECK_EQUAL(string, "bla");

    string = geoneric::String("bla bla").strip("ba");
    BOOST_CHECK_EQUAL(string, "la bl");

    string = geoneric::String("").strip();
    BOOST_CHECK_EQUAL(string, "");

    string = geoneric::String("").strip("");
    BOOST_CHECK_EQUAL(string, "");

    string = geoneric::String("øaø").strip("ø");
    BOOST_CHECK_EQUAL(string, "a");
}


BOOST_AUTO_TEST_CASE(replace)
{
    geoneric::String string;

    string = geoneric::String("<string>").replace("<", "&lt;");
    BOOST_CHECK_EQUAL(string, "&lt;string>");
}

BOOST_AUTO_TEST_SUITE_END()
