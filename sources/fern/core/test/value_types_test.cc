#define BOOST_TEST_MODULE fern operation_core
#include <boost/test/unit_test.hpp>
#include "fern/core/value_types.h"


BOOST_AUTO_TEST_SUITE(value_types)

BOOST_AUTO_TEST_CASE(string)
{
    BOOST_CHECK_EQUAL(fern::ValueTypes::UNKNOWN.to_string(), "?");
    BOOST_CHECK_EQUAL(fern::ValueTypes::UINT8.to_string(), "Uint8");
    BOOST_CHECK_EQUAL(fern::ValueTypes::INT64.to_string(), "Int64");


    fern::ValueTypes value_types(fern::ValueTypes::UINT8);
    BOOST_CHECK_EQUAL(value_types.to_string(), "Uint8");
}


BOOST_AUTO_TEST_CASE(unknown)
{
    fern::ValueTypes value_types;

    BOOST_CHECK_EQUAL(value_types, fern::ValueTypes::UNKNOWN);
}

BOOST_AUTO_TEST_SUITE_END()
