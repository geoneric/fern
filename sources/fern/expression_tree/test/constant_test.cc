#define BOOST_TEST_MODULE fern expression_tree constant
#include <boost/test/unit_test.hpp>
#include "fern/expression_tree/constant.h"


BOOST_AUTO_TEST_SUITE(constant)

BOOST_AUTO_TEST_CASE(use_cases)
{
    {
        fern::Constant<int32_t> constant(2);

        BOOST_CHECK_EQUAL(static_cast<int>(constant), 2);
        BOOST_CHECK_EQUAL(int(constant), 2);
    }
}

BOOST_AUTO_TEST_SUITE_END()
