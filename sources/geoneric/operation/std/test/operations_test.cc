#define BOOST_TEST_MODULE geoneric operation_std
#include <boost/test/unit_test.hpp>
#include "geoneric/operation/std/operations.h"


BOOST_AUTO_TEST_SUITE(operations)

BOOST_AUTO_TEST_CASE(operations)
{
    BOOST_CHECK( geoneric::operations()->has_operation("abs"));
    BOOST_CHECK(!geoneric::operations()->has_operation("sba"));
}

BOOST_AUTO_TEST_SUITE_END()

