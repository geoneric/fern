// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern core
#include <boost/test/unit_test.hpp>
#include "fern/core/scope_exit.h"


BOOST_AUTO_TEST_SUITE(scope_exit)

BOOST_AUTO_TEST_CASE(general)
{
    int a;

    // Let the instance call its function.
    a = 5;
    {
        auto set_a_to_2 = fern::makeScopeExit([&a]() { a = 2; });
    }
    BOOST_CHECK_EQUAL(a, 2);


    // Release the instance of it responsibility.
    a = 5;
    {
        auto set_a_to_2 = fern::makeScopeExit([&a]() { a = 2; });
        set_a_to_2.release();
    }
    BOOST_CHECK_EQUAL(a, 5);
}

BOOST_AUTO_TEST_SUITE_END()
