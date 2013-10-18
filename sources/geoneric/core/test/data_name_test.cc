#define BOOST_TEST_MODULE geoneric core
#include <boost/test/unit_test.hpp>
#include "geoneric/core/data_name.h"


BOOST_AUTO_TEST_SUITE(data_name)

BOOST_AUTO_TEST_CASE(constructor)
{
    {
        geoneric::DataName name("dataset:path");
        BOOST_CHECK_EQUAL(name.database_pathname(), geoneric::Path("dataset"));
        BOOST_CHECK_EQUAL(name.data_pathname(), geoneric::Path("path"));
    }

    {
        geoneric::DataName name("dataset");
        BOOST_CHECK_EQUAL(name.database_pathname(), geoneric::Path("dataset"));
        BOOST_CHECK_EQUAL(name.data_pathname(), geoneric::Path(""));
    }

    {
        geoneric::DataName name("dataset:");
        BOOST_CHECK_EQUAL(name.database_pathname(), geoneric::Path("dataset"));
        BOOST_CHECK_EQUAL(name.data_pathname(), geoneric::Path(""));
    }

    {
        geoneric::DataName name("dataset:path:to");
        BOOST_CHECK_EQUAL(name.database_pathname(), geoneric::Path("dataset"));
        BOOST_CHECK_EQUAL(name.data_pathname(), geoneric::Path("path:to"));
    }

    {
        geoneric::DataName name("dataset:path/to");
        BOOST_CHECK_EQUAL(name.database_pathname(), geoneric::Path("dataset"));
        BOOST_CHECK_EQUAL(name.data_pathname(), geoneric::Path("path/to"));
    }

    {
        geoneric::DataName name("dataset:path:to");
        BOOST_CHECK_EQUAL(name.database_pathname(), geoneric::Path("dataset"));
        BOOST_CHECK_EQUAL(name.data_pathname(), geoneric::Path("path:to"));
    }

    {
        geoneric::DataName name("dataset:path/////to//");
        BOOST_CHECK_EQUAL(name.database_pathname(), geoneric::Path("dataset"));
        BOOST_CHECK_EQUAL(name.data_pathname(), geoneric::Path("path/to"));
    }
}

BOOST_AUTO_TEST_SUITE_END()
