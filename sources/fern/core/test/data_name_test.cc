#define BOOST_TEST_MODULE geoneric core
#include <boost/test/unit_test.hpp>
#include "fern/core/data_name.h"


BOOST_AUTO_TEST_SUITE(data_name)

BOOST_AUTO_TEST_CASE(constructor)
{
    using namespace fern;

    {
        DataName name("dataset:path");
        BOOST_CHECK_EQUAL(String(name.database_pathname()), "dataset");
        BOOST_CHECK_EQUAL(String(name.data_pathname()), "/path");
    }

    {
        DataName name("dataset:/path");
        BOOST_CHECK_EQUAL(String(name.database_pathname()), "dataset");
        BOOST_CHECK_EQUAL(String(name.data_pathname()), "/path");
    }

    {
        DataName name("dataset");
        BOOST_CHECK_EQUAL(String(name.database_pathname()), "dataset");
        BOOST_CHECK_EQUAL(String(name.data_pathname()), "/");
    }

    {
        DataName name("dataset:");
        BOOST_CHECK_EQUAL(String(name.database_pathname()), "dataset");
        BOOST_CHECK_EQUAL(String(name.data_pathname()), "/");
    }

    {
        DataName name("dataset:/");
        BOOST_CHECK_EQUAL(String(name.database_pathname()), "dataset");
        BOOST_CHECK_EQUAL(String(name.data_pathname()), "/");
    }

    {
        DataName name("dataset:path:to");
        BOOST_CHECK_EQUAL(String(name.database_pathname()), "dataset");
        BOOST_CHECK_EQUAL(String(name.data_pathname()), "/path:to");
    }

    {
        DataName name("dataset:/path:to");
        BOOST_CHECK_EQUAL(String(name.database_pathname()), "dataset");
        BOOST_CHECK_EQUAL(String(name.data_pathname()), "/path:to");
    }

    {
        DataName name("dataset:path/to");
        BOOST_CHECK_EQUAL(String(name.database_pathname()), "dataset");
        BOOST_CHECK_EQUAL(String(name.data_pathname()), "/path/to");
    }

    {
        DataName name("dataset:/path/to");
        BOOST_CHECK_EQUAL(String(name.database_pathname()), "dataset");
        BOOST_CHECK_EQUAL(String(name.data_pathname()), "/path/to");
    }

    {
        DataName name("dataset:path/////to//");
        BOOST_CHECK_EQUAL(String(name.database_pathname()), "dataset");
        BOOST_CHECK_EQUAL(String(name.data_pathname()), "/path/to");
    }

    {
        DataName name("dataset:///path/////to//");
        BOOST_CHECK_EQUAL(String(name.database_pathname()), "dataset");
        BOOST_CHECK_EQUAL(String(name.data_pathname()), "/path/to");
    }
}

BOOST_AUTO_TEST_SUITE_END()
