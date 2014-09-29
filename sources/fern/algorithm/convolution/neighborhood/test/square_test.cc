#define BOOST_TEST_MODULE fern algorithm convolution neighborhood square
#include <boost/test/unit_test.hpp>
#include "fern/core/constant_traits.h"
#include "fern/algorithm/convolution/neighborhood/square_traits.h"
#include "fern/algorithm/algebra/elementary/multiply.h"


namespace fa = fern::algorithm;


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


BOOST_AUTO_TEST_CASE(verify_use_in_algorithm)
{
    // Verify we can pass a square to an operation.

    // Multiply.
    {
        fern::Square<int, 1> square({
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
        });

        fa::algebra::multiply(fa::sequential, square, 2, square);

        BOOST_CHECK_EQUAL(square[0][0], 2);
        BOOST_CHECK_EQUAL(square[0][1], 4);
        BOOST_CHECK_EQUAL(square[0][2], 6);
        BOOST_CHECK_EQUAL(square[1][0], 8);
        BOOST_CHECK_EQUAL(square[1][1], 10);
        BOOST_CHECK_EQUAL(square[1][2], 12);
        BOOST_CHECK_EQUAL(square[2][0], 14);
        BOOST_CHECK_EQUAL(square[2][1], 16);
        BOOST_CHECK_EQUAL(square[2][2], 18);
    }
}

BOOST_AUTO_TEST_SUITE_END()
