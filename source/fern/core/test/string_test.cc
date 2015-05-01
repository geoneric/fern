// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern core
#include <boost/test/unit_test.hpp>
#include "fern/core/string.h"


BOOST_AUTO_TEST_SUITE(string)

BOOST_AUTO_TEST_CASE(constructor)
{
    std::string string = "aæø";
    BOOST_CHECK_EQUAL(string, "aæø");
}


BOOST_AUTO_TEST_CASE(strip)
{
    std::string string;

    string = "bla";
    string = fern::strip(string);
    BOOST_CHECK_EQUAL(string, "bla");

    string = "bla";
    string = fern::strip(string, "\n");
    BOOST_CHECK_EQUAL(string, "bla");

    string = "bla\n";
    string = fern::strip(string);
    BOOST_CHECK_EQUAL(string, "bla");

    string = "bla\n";
    string = fern::strip(string, "\n");
    BOOST_CHECK_EQUAL(string, "bla");

    string = "\nbla";
    string = fern::strip(string);
    BOOST_CHECK_EQUAL(string, "bla");

    string = "\nbla";
    string = fern::strip(string, "\n");
    BOOST_CHECK_EQUAL(string, "bla");

    string = "\nbla\n";
    string = fern::strip(string);
    BOOST_CHECK_EQUAL(string, "bla");

    string = "\nbla\n";
    string = fern::strip(string, "\n");
    BOOST_CHECK_EQUAL(string, "bla");

    // Strip with argument preserves whitespace.
    string = "\n bla \n";
    string = fern::strip(string, "\n");
    BOOST_CHECK_EQUAL(string, " bla ");

    // Default strip doesn't preserve whitespace.
    string = "\n bla \n";
    string = fern::strip(string);
    BOOST_CHECK_EQUAL(string, "bla");

    string = "bla bla";
    string = fern::strip(string, "ba");
    BOOST_CHECK_EQUAL(string, "la bl");

    string = "";
    string = fern::strip(string);
    BOOST_CHECK_EQUAL(string, "");

    string = "";
    string = fern::strip(string, "");
    BOOST_CHECK_EQUAL(string, "");

    string = "";
    string = fern::strip(string, "x");
    BOOST_CHECK_EQUAL(string, "");

    string = "øaø";
    string = fern::strip(string, "ø");
    BOOST_CHECK_EQUAL(string, "a");
}


BOOST_AUTO_TEST_CASE(replace)
{
    std::string string;

    string = "<string>";
    string = fern::replace(string, "<", "&lt;");
    BOOST_CHECK_EQUAL(string, "&lt;string>");

    string = "//";
    string = fern::replace(string, "//", "/");
    BOOST_CHECK_EQUAL(string, "/");

    string = "///";
    string = fern::replace(string, "//", "/");
    BOOST_CHECK_EQUAL(string, "//");

    string = "////";
    string = fern::replace(string, "//", "/");
    BOOST_CHECK_EQUAL(string, "//");
}


BOOST_AUTO_TEST_CASE(contains)
{
    BOOST_CHECK( fern::contains("<string>", "<"));
    BOOST_CHECK(!fern::contains("string>", "<"));
}


#ifndef FERN_COMPILER_DOES_NOT_HAVE_REGEX
BOOST_AUTO_TEST_CASE(split_)
{
    std::string string;
    std::vector<std::string> words;

    string = std::string();
    words = fern::split(string, "");
    BOOST_CHECK(words.empty());

    string = std::string();
    words = fern::split(string, "x");
    BOOST_CHECK(words.empty());

    string = std::string("abxcd");
    words = fern::split(string, "x");
    BOOST_REQUIRE_EQUAL(words.size(), 2u);
    BOOST_CHECK_EQUAL(words[0], std::string("ab"));
    BOOST_CHECK_EQUAL(words[1], std::string("cd"));

    string = std::string("xxabxxcdxx");
    words = fern::split(string, "x");
    BOOST_REQUIRE_EQUAL(words.size(), 2u);
    BOOST_CHECK_EQUAL(words[0], std::string("ab"));
    BOOST_CHECK_EQUAL(words[1], std::string("cd"));

    string = std::string("a,b,c");
    words = fern::split(string, ",");
    BOOST_REQUIRE_EQUAL(words.size(), 3u);
    BOOST_CHECK_EQUAL(words[0], std::string("a"));
    BOOST_CHECK_EQUAL(words[1], std::string("b"));
    BOOST_CHECK_EQUAL(words[2], std::string("c"));

    string = std::string(" a, b ,c ");
    words = fern::split(string, ",");
    BOOST_REQUIRE_EQUAL(words.size(), 3u);
    BOOST_CHECK_EQUAL(words[0], std::string(" a"));
    BOOST_CHECK_EQUAL(words[1], std::string(" b "));
    BOOST_CHECK_EQUAL(words[2], std::string("c "));
}
#endif


// BOOST_AUTO_TEST_CASE(encode_in_default_encoding_)
// {
//     fern::String string;
// 
//     string = "";
//     BOOST_CHECK_EQUAL(encode_in_default_encoding(string), std::string(""));
// }


// BOOST_AUTO_TEST_CASE(encode_in_utf8_)
// {
//     fern::String string;
// 
//     string = "";
//     BOOST_CHECK_EQUAL(encode_in_utf8(string), std::string(""));
// }

BOOST_AUTO_TEST_SUITE_END()
