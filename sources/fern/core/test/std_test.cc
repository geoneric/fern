#define BOOST_TEST_MODULE fern core
#include <vector>
#include <boost/test/unit_test.hpp>
#include "fern/core/std.h"


BOOST_AUTO_TEST_SUITE(std_)

BOOST_AUTO_TEST_CASE(sort)
{
    using Container = std::vector<int>;

    {
        Container container{5, 3, 4, 1, 2};
        // Descending is default.
        fern::sort(container);
        BOOST_CHECK(container == (Container{1, 2, 3, 4, 5}));
    }

    {
        Container container{5, 3, 4, 1, 2};
        // Now ascending.
        fern::sort(container, [](int lhs, int rhs){ return lhs > rhs; });
        BOOST_CHECK(container == (Container{5, 4, 3, 2, 1}));
    }
}

BOOST_AUTO_TEST_SUITE_END()
