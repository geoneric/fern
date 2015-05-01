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
#include "fern/core/data_name.h"


BOOST_AUTO_TEST_SUITE(data_name)

BOOST_AUTO_TEST_CASE(constructor)
{
    using namespace fern;

    {
        DataName name1("blaah");
        DataName name2(std::string("blaah"));
        DataName name3(name2);
    }

    {
        DataName name("dataset:path");
        BOOST_CHECK_EQUAL(name.database_pathname(), Path("dataset"));
        BOOST_CHECK_EQUAL(name.data_pathname(), Path("/path"));
    }

    {
        DataName name("dataset:/path");
        BOOST_CHECK_EQUAL(name.database_pathname(), Path("dataset"));
        BOOST_CHECK_EQUAL(name.data_pathname(), Path("/path"));
    }

    {
        DataName name("dataset");
        BOOST_CHECK_EQUAL(name.database_pathname(), Path("dataset"));
        BOOST_CHECK_EQUAL(name.data_pathname(), Path("/"));
    }

    {
        DataName name("dataset:");
        BOOST_CHECK_EQUAL(name.database_pathname(), Path("dataset"));
        BOOST_CHECK_EQUAL(name.data_pathname(), Path("/"));
    }

    {
        DataName name("dataset:/");
        BOOST_CHECK_EQUAL(name.database_pathname(), Path("dataset"));
        BOOST_CHECK_EQUAL(name.data_pathname(), Path("/"));
    }

    {
        DataName name("dataset:path:to");
        BOOST_CHECK_EQUAL(name.database_pathname(), Path("dataset"));
        BOOST_CHECK_EQUAL(name.data_pathname(), Path("/path:to"));
    }

    {
        DataName name("dataset:/path:to");
        BOOST_CHECK_EQUAL(name.database_pathname(), Path("dataset"));
        BOOST_CHECK_EQUAL(name.data_pathname(), Path("/path:to"));
    }

    {
        DataName name("dataset:path/to");
        BOOST_CHECK_EQUAL(name.database_pathname(), Path("dataset"));
        BOOST_CHECK_EQUAL(name.data_pathname(), Path("/path/to"));
    }

    {
        DataName name("dataset:/path/to");
        BOOST_CHECK_EQUAL(name.database_pathname(), Path("dataset"));
        BOOST_CHECK_EQUAL(name.data_pathname(), Path("/path/to"));
    }

    {
        DataName name("dataset:path/////to//");
        BOOST_CHECK_EQUAL(name.database_pathname(), Path("dataset"));
        BOOST_CHECK_EQUAL(name.data_pathname(), Path("/path/to"));
    }

    {
        DataName name("dataset:///path/////to//");
        BOOST_CHECK_EQUAL(name.database_pathname(), Path("dataset"));
        BOOST_CHECK_EQUAL(name.data_pathname(), Path("/path/to"));
    }
}


BOOST_AUTO_TEST_CASE(assignment)
{
    using namespace fern;

    {
        DataName name1 = "blaah";
        DataName name2 = std::string("blaah");
        DataName name3 = name2;
    }
}

BOOST_AUTO_TEST_SUITE_END()
