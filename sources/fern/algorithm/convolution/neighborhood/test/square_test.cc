#define BOOST_TEST_MODULE fern algorithm convolution neighborhood square
#include <boost/test/unit_test.hpp>
#include "fern/algorithm/convolution/neighborhood/square.h"


BOOST_AUTO_TEST_SUITE(square)

BOOST_AUTO_TEST_CASE(square)
{
    {
        fern::Square<int, 1> square({
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
        });

        BOOST_CHECK_EQUAL(square[0][0], 1);
        BOOST_CHECK_EQUAL(square[1][1], 5);
        BOOST_CHECK_EQUAL(square[2][2], 9);
    }
}

BOOST_AUTO_TEST_SUITE_END()
