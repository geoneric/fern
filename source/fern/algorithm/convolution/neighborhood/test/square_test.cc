// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern algorithm convolution neighborhood square
#include <boost/test/unit_test.hpp>
#include "fern/core/data_customization_point/scalar.h"
#include "fern/algorithm/convolution/neighborhood/square_traits.h"
#include "fern/algorithm/algebra/elementary/multiply.h"


namespace fa = fern::algorithm;


BOOST_AUTO_TEST_CASE(square)
{
    {
        fern::Square<int, 1> square({
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
        });

        BOOST_CHECK_EQUAL(square.weight(0), 1);
        BOOST_CHECK_EQUAL(square.weight(4), 5);
        BOOST_CHECK_EQUAL(square.weight(8), 9);
    }
}


BOOST_AUTO_TEST_CASE(verify_use_in_algorithm)
{
    // Verify we can pass a square to an operation.

    fa::SequentialExecutionPolicy sequential;

    // Multiply.
    {
        fern::Square<int, 1> square({
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
        });

        fa::algebra::multiply(sequential, square, 2, square);

        BOOST_CHECK_EQUAL(square.weight(0), 2);
        BOOST_CHECK_EQUAL(square.weight(1), 4);
        BOOST_CHECK_EQUAL(square.weight(2), 6);
        BOOST_CHECK_EQUAL(square.weight(3), 8);
        BOOST_CHECK_EQUAL(square.weight(4), 10);
        BOOST_CHECK_EQUAL(square.weight(5), 12);
        BOOST_CHECK_EQUAL(square.weight(6), 14);
        BOOST_CHECK_EQUAL(square.weight(7), 16);
        BOOST_CHECK_EQUAL(square.weight(8), 18);
    }
}
