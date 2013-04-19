#define BOOST_TEST_MODULE ranally operation_std
#include <boost/test/unit_test.hpp>
#include "ranally/operation/std/operations.h"


BOOST_AUTO_TEST_SUITE(operations)

BOOST_AUTO_TEST_CASE(operations)
{
    BOOST_CHECK( ranally::operations->has_operation("abs"));
    BOOST_CHECK(!ranally::operations->has_operation("sba"));
}

BOOST_AUTO_TEST_SUITE_END()

