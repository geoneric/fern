#define BOOST_TEST_MODULE geoneric operation_core
#include <boost/test/unit_test.hpp>
#include "geoneric/operation/core/result_type.h"


BOOST_AUTO_TEST_SUITE(result_type)

BOOST_AUTO_TEST_CASE(result_type)
{
    geoneric::ResultType result_type(geoneric::DataTypes::CONSTANT,
        geoneric::ValueTypes::INT32);
    BOOST_CHECK_EQUAL(result_type.data_type(), geoneric::DataTypes::CONSTANT);
    BOOST_CHECK_EQUAL(result_type.value_type(), geoneric::ValueTypes::INT32);

    {
        geoneric::ResultType copy(result_type);
        BOOST_CHECK_EQUAL(result_type.data_type(),
            geoneric::DataTypes::CONSTANT);
        BOOST_CHECK_EQUAL(result_type.value_type(),
            geoneric::ValueTypes::INT32);
    }

    {
        geoneric::ResultType copy = result_type;
        BOOST_CHECK_EQUAL(result_type.data_type(),
            geoneric::DataTypes::CONSTANT);
        BOOST_CHECK_EQUAL(result_type.value_type(), geoneric::ValueTypes::INT32);
    }
}

BOOST_AUTO_TEST_SUITE_END()
