#define BOOST_TEST_MODULE geoneric io
#include <boost/test/unit_test.hpp>
#include "geoneric/io/gdal/drivers.h"


BOOST_AUTO_TEST_SUITE(drivers)

BOOST_AUTO_TEST_CASE(drivers)
{
    geoneric::String name = "raster-1.asc";
    auto dataset = geoneric::open(name);
    BOOST_REQUIRE(dataset);
}

BOOST_AUTO_TEST_SUITE_END()
