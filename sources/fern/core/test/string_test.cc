#define BOOST_TEST_MODULE fern core
#include <boost/test/unit_test.hpp>
#include "fern/core/string.h"


BOOST_AUTO_TEST_SUITE(string)

BOOST_AUTO_TEST_CASE(constructor)
{
    fern::String string;

    string = "aæø";
    BOOST_CHECK_EQUAL(string.encode_in_utf8(), "aæø");
}


BOOST_AUTO_TEST_CASE(strip)
{
    fern::String string;

    string = fern::String("bla").strip();
    BOOST_CHECK_EQUAL(string, "bla");

    string = fern::String("bla").strip("\n");
    BOOST_CHECK_EQUAL(string, "bla");

    string = fern::String("bla\n").strip();
    BOOST_CHECK_EQUAL(string, "bla");

    string = fern::String("bla\n").strip("\n");
    BOOST_CHECK_EQUAL(string, "bla");

    string = fern::String("\nbla").strip();
    BOOST_CHECK_EQUAL(string, "bla");

    string = fern::String("\nbla").strip("\n");
    BOOST_CHECK_EQUAL(string, "bla");

    string = fern::String("\nbla\n").strip();
    BOOST_CHECK_EQUAL(string, "bla");

    string = fern::String("\nbla\n").strip("\n");
    BOOST_CHECK_EQUAL(string, "bla");

    // Strip with argument preserves whitespace.
    string = fern::String("\n bla \n").strip("\n");
    BOOST_CHECK_EQUAL(string, " bla ");

    // Default strip doesn't preserve whitespace.
    string = fern::String("\n bla \n").strip();
    BOOST_CHECK_EQUAL(string, "bla");

    string = fern::String("bla bla").strip("ba");
    BOOST_CHECK_EQUAL(string, "la bl");

    string = fern::String("").strip();
    BOOST_CHECK_EQUAL(string, "");

    string = fern::String("").strip("");
    BOOST_CHECK_EQUAL(string, "");

    string = fern::String("").strip("x");
    BOOST_CHECK_EQUAL(string, "");

    string = fern::String("øaø").strip("ø");
    BOOST_CHECK_EQUAL(string, "a");
}


BOOST_AUTO_TEST_CASE(replace)
{
    fern::String string;

    string = fern::String("<string>").replace("<", "&lt;");
    BOOST_CHECK_EQUAL(string, "&lt;string>");

    string = fern::String("//").replace("//", "/");
    BOOST_CHECK_EQUAL(string, "/");

    string = fern::String("///").replace("//", "/");
    BOOST_CHECK_EQUAL(string, "//");

    string = fern::String("////").replace("//", "/");
    BOOST_CHECK_EQUAL(string, "//");
}


BOOST_AUTO_TEST_CASE(contains)
{
    BOOST_CHECK( fern::String("<string>").contains("<"));
    BOOST_CHECK(!fern::String("string>").contains("<"));
}


BOOST_AUTO_TEST_CASE(split)
{
    fern::String string;
    std::vector<fern::String> words;

    string = fern::String();
    words = string.split("");
    BOOST_CHECK(words.empty());

    string = fern::String();
    words = string.split("x");
    BOOST_CHECK(words.empty());

    string = fern::String("abxcd");
    words = string.split("x+");
    BOOST_REQUIRE_EQUAL(words.size(), 2u);
    BOOST_CHECK_EQUAL(words[0], "ab");
    BOOST_CHECK_EQUAL(words[1], "cd");

    string = fern::String("xxabxxcdxx");
    words = string.split("x+");
    BOOST_REQUIRE_EQUAL(words.size(), 2u);
    BOOST_CHECK_EQUAL(words[0], "ab");
    BOOST_CHECK_EQUAL(words[1], "cd");
}


BOOST_AUTO_TEST_CASE(encode_in_default_encoding)
{
    fern::String string;

    string = "";
    BOOST_CHECK_EQUAL(string.encode_in_default_encoding(), std::string(""));
}


BOOST_AUTO_TEST_CASE(encode_in_utf8)
{
    fern::String string;

    string = "";
    BOOST_CHECK_EQUAL(string.encode_in_utf8(), std::string(""));
}

BOOST_AUTO_TEST_SUITE_END()