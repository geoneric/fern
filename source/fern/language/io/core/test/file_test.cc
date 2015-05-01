// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern io
#include <boost/test/unit_test.hpp>
#include "fern/language/io/core/file.h"


namespace fl = fern::language;


BOOST_AUTO_TEST_SUITE(file)

BOOST_AUTO_TEST_CASE(file)
{
    BOOST_CHECK( fl::file_exists("raster-1.asc"));
    BOOST_CHECK(!fl::file_exists("does_not_exist.asc"));
    BOOST_CHECK( fl::file_exists("write_only.asc"));
    BOOST_CHECK( fl::file_exists("raster-1-link.asc"));
    BOOST_CHECK(!fl::file_exists("raster-1-dangling_link.asc"));
}

BOOST_AUTO_TEST_SUITE_END()
