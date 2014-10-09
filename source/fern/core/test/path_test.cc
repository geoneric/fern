#define BOOST_TEST_MODULE fern core
#include <boost/test/unit_test.hpp>
#include "fern/core/path.h"


BOOST_AUTO_TEST_SUITE(path)

BOOST_AUTO_TEST_CASE(names)
{
    fern::Path path;
    std::vector<fern::String> names;

    path = "a/b/c";
    names = path.names();
    BOOST_REQUIRE_EQUAL(names.size(), 3u);
    BOOST_CHECK_EQUAL(names[0], fern::String("a"));
    BOOST_CHECK_EQUAL(names[1], fern::String("b"));
    BOOST_CHECK_EQUAL(names[2], fern::String("c"));

    path = "/a/b/c/";
    names = path.names();
    BOOST_REQUIRE_EQUAL(names.size(), 3u);
    BOOST_CHECK_EQUAL(names[0], fern::String("a"));
    BOOST_CHECK_EQUAL(names[1], fern::String("b"));
    BOOST_CHECK_EQUAL(names[2], fern::String("c"));

    path = "/";
    names = path.names();
    BOOST_CHECK(names.empty());

    path = "//";
    names = path.names();
    BOOST_CHECK(names.empty());

    path = "/\\/";
    names = path.names();
    BOOST_REQUIRE_EQUAL(names.size(), 1u);
    BOOST_CHECK_EQUAL(names[0], fern::String("\\"));
}


BOOST_AUTO_TEST_SUITE_END()
