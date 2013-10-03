#define BOOST_TEST_MODULE geoneric io
#include <boost/test/unit_test.hpp>
#include "geoneric/io/core/file.h"


BOOST_AUTO_TEST_SUITE(file)

BOOST_AUTO_TEST_CASE(file)
{
    BOOST_CHECK( geoneric::file_exists("raster-1.asc"));
    BOOST_CHECK(!geoneric::file_exists("does_not_exist.asc"));
    BOOST_CHECK( geoneric::file_exists("write_only.asc"));
    BOOST_CHECK( geoneric::file_exists("raster-1-link.asc"));
    BOOST_CHECK(!geoneric::file_exists("raster-1-dangling_link.asc"));
}

BOOST_AUTO_TEST_SUITE_END()
