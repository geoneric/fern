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
#include "fern/core/path.h"


BOOST_AUTO_TEST_SUITE(path)

BOOST_AUTO_TEST_CASE(construction)
{
    fern::Path p1("blaah");
    fern::Path p2(std::string("blaah"));
    fern::Path p3(p2);
}


BOOST_AUTO_TEST_CASE(assignment)
{
    fern::Path p1 = "blaah";
    fern::Path p2 = std::string("blaah");
    fern::Path p3 = p2;
}


BOOST_AUTO_TEST_CASE(names)
{
    fern::Path path;
    std::vector<std::string> names;

    path = "a/b/c";
    names = path.names();
    BOOST_REQUIRE_EQUAL(names.size(), 3u);
    BOOST_CHECK_EQUAL(names[0], "a");
    BOOST_CHECK_EQUAL(names[1], "b");
    BOOST_CHECK_EQUAL(names[2], "c");

    path = "/a/b/c/";
    names = path.names();
    BOOST_REQUIRE_EQUAL(names.size(), 3u);
    BOOST_CHECK_EQUAL(names[0], "a");
    BOOST_CHECK_EQUAL(names[1], "b");
    BOOST_CHECK_EQUAL(names[2], "c");

    path = "/";
    names = path.names();
    BOOST_CHECK(names.empty());

    path = "//";
    names = path.names();
    BOOST_CHECK(names.empty());

    path = "/\\/";
    names = path.names();
    BOOST_REQUIRE_EQUAL(names.size(), 1u);
    BOOST_CHECK_EQUAL(names[0], "\\");
}


BOOST_AUTO_TEST_SUITE_END()
