// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern compiler
#include <boost/test/unit_test.hpp>
#include "fern/language/compiler/compiler.h"


BOOST_AUTO_TEST_SUITE(compiler)

BOOST_AUTO_TEST_CASE(constructor)
{
    fern::Compiler compiler("h", "cc");
}

BOOST_AUTO_TEST_SUITE_END()
