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
    fern::String string;

    string = "aæø";
    BOOST_CHECK_EQUAL(string.encode_in_utf8(), "aæø");
}


BOOST_AUTO_TEST_CASE(strip)
{
    fern::String string;

    string = "bla";
    string = fern::strip(string);
    BOOST_CHECK_EQUAL(string, fern::String("bla"));

    string = "bla";
    string = fern::strip(string, "\n");
    BOOST_CHECK_EQUAL(string, fern::String("bla"));

    string = "bla\n";
    string = fern::strip(string);
    BOOST_CHECK_EQUAL(string, fern::String("bla"));

    string = "bla\n";
    string = fern::strip(string, "\n");
    BOOST_CHECK_EQUAL(string, fern::String("bla"));

    string = "\nbla";
    string = fern::strip(string);
    BOOST_CHECK_EQUAL(string, fern::String("bla"));

    string = "\nbla";
    string = fern::strip(string, "\n");
    BOOST_CHECK_EQUAL(string, fern::String("bla"));

    string = "\nbla\n";
    string = fern::strip(string);
    BOOST_CHECK_EQUAL(string, fern::String("bla"));

    string = "\nbla\n";
    string = fern::strip(string, "\n");
    BOOST_CHECK_EQUAL(string, fern::String("bla"));

    // Strip with argument preserves whitespace.
    string = "\n bla \n";
    string = fern::strip(string, "\n");
    BOOST_CHECK_EQUAL(string, fern::String(" bla "));

    // Default strip doesn't preserve whitespace.
    string = "\n bla \n";
    string = fern::strip(string);
    BOOST_CHECK_EQUAL(string, fern::String("bla"));

    string = "bla bla";
    string = fern::strip(string, "ba");
    BOOST_CHECK_EQUAL(string, fern::String("la bl"));

    string = "";
    string = fern::strip(string);
    BOOST_CHECK_EQUAL(string, fern::String(""));

    string = "";
    string = fern::strip(string, "");
    BOOST_CHECK_EQUAL(string, fern::String(""));

    string = "";
    string = fern::strip(string, "x");
    BOOST_CHECK_EQUAL(string, fern::String(""));

    string = "øaø";
    string = fern::strip(string, "ø");
    BOOST_CHECK_EQUAL(string, fern::String("a"));
}


BOOST_AUTO_TEST_CASE(replace)
{
    fern::String string;

    string = fern::String("<string>").replace("<", "&lt;");
    BOOST_CHECK_EQUAL(string, fern::String("&lt;string>"));

    string = fern::String("//").replace("//", "/");
    BOOST_CHECK_EQUAL(string, fern::String("/"));

    string = fern::String("///").replace("//", "/");
    BOOST_CHECK_EQUAL(string, fern::String("//"));

    string = fern::String("////").replace("//", "/");
    BOOST_CHECK_EQUAL(string, fern::String("//"));
}


BOOST_AUTO_TEST_CASE(contains)
{
    BOOST_CHECK( fern::String("<string>").contains("<"));
    BOOST_CHECK(!fern::String("string>").contains("<"));
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
