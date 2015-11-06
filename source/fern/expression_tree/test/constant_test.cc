// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern expression_tree constant
#include <boost/test/unit_test.hpp>
#include "fern/expression_tree/constant.h"


BOOST_AUTO_TEST_CASE(use_cases)
{
    {
        fern::expression_tree::Constant<int32_t> constant(2);

        BOOST_CHECK_EQUAL(static_cast<int>(constant), 2);
        BOOST_CHECK_EQUAL(int(constant), 2);
    }
}
