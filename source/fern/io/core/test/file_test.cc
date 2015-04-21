// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern io core file
#include <boost/test/unit_test.hpp>
#include "fern/io/core/file.h"


namespace fi = fern::io;


BOOST_AUTO_TEST_SUITE(file)

BOOST_AUTO_TEST_CASE(file_exists)
{
    BOOST_CHECK(!fi::file_exists("does_not_exist.txt"));
    BOOST_CHECK( fi::file_exists("unreadable.txt"));
    BOOST_CHECK( fi::file_exists("unwritable.txt"));
}

BOOST_AUTO_TEST_SUITE_END()
