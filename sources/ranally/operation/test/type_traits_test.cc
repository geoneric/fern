#define BOOST_TEST_MODULE ranally operation
#include <boost/test/unit_test.hpp>
#include "ranally/operation/type_traits.h"


BOOST_AUTO_TEST_SUITE(result_type)

BOOST_AUTO_TEST_CASE(result_type)
{
    BOOST_CHECK_EQUAL(ranally::TypeTraits<uint8_t>::value_type,
        ranally::ValueTypes::UINT8);
}

BOOST_AUTO_TEST_SUITE_END()
