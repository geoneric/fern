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
#include "fern/core/stopwatch.h"


void my_function()
{
    for(size_t i = 0; i < 1000000; ++i) {
        std::sqrt(123.456L);
    }

    return;
}


struct MyStruct
{
    void operator()()
    {
        my_function();
    }
};


BOOST_AUTO_TEST_SUITE(stopwatch)

BOOST_AUTO_TEST_CASE(constructor)
{
    {
        fern::Stopwatch stopwatch(my_function);
        /// BOOST_CHECK(stopwatch.clock_ticks() > 0);
        BOOST_CHECK(stopwatch.wall_time() > 0);
        // Fails sometimes BOOST_CHECK(stopwatch.user_time() > 0);
        BOOST_CHECK_EQUAL(stopwatch.system_time(), 0);
    }

    {
        fern::Stopwatch stopwatch((MyStruct()));
        /// BOOST_CHECK(stopwatch.clock_ticks() > 0);
        BOOST_CHECK(stopwatch.wall_time() > 0);
        // Fails sometimes BOOST_CHECK(stopwatch.user_time() > 0);
        BOOST_CHECK_EQUAL(stopwatch.system_time(), 0);
    }
}

BOOST_AUTO_TEST_SUITE_END()
