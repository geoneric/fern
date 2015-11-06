// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern operation std operations
#include <boost/test/unit_test.hpp>
#include "fern/language/operation/std/operations.h"


namespace fl = fern::language;


BOOST_AUTO_TEST_CASE(operations)
{
    BOOST_CHECK( fl::operations()->has_operation("abs"));
    BOOST_CHECK(!fl::operations()->has_operation("sba"));
}
