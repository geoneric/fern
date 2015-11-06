// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern core std
#include <vector>
#include <boost/test/unit_test.hpp>
#include "fern/core/std.h"


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
