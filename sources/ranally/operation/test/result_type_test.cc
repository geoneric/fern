#define BOOST_TEST_MODULE ranally operation
#include <boost/test/unit_test.hpp>
#include "ranally/operation/result_type.h"


BOOST_AUTO_TEST_SUITE(result_type)

BOOST_AUTO_TEST_CASE(result_type)
{
    ranally::ResultType result_type(ranally::DataTypes::SCALAR,
        ranally::ValueTypes::INT32);
    BOOST_CHECK_EQUAL(result_type.data_type(), ranally::DataTypes::SCALAR);
    BOOST_CHECK_EQUAL(result_type.value_type(), ranally::ValueTypes::INT32);

    {
        ranally::ResultType copy(result_type);
        BOOST_CHECK_EQUAL(result_type.data_type(), ranally::DataTypes::SCALAR);
        BOOST_CHECK_EQUAL(result_type.value_type(), ranally::ValueTypes::INT32);
    }

    {
        ranally::ResultType copy = result_type;
        BOOST_CHECK_EQUAL(result_type.data_type(), ranally::DataTypes::SCALAR);
        BOOST_CHECK_EQUAL(result_type.value_type(), ranally::ValueTypes::INT32);
    }
}

BOOST_AUTO_TEST_SUITE_END()
