#define BOOST_TEST_MODULE geoneric io
#include <boost/test/unit_test.hpp>
#include "geoneric/io/gdal/drivers.h"
#include "geoneric/io/gdal/gdal_client.h"


class Support:
    public geoneric::GDALClient
{

public:

    Support()
        : geoneric::GDALClient()
    {
    }

};


BOOST_FIXTURE_TEST_SUITE(drivers, Support)

BOOST_AUTO_TEST_CASE(drivers)
{
    geoneric::String name = "raster-1.asc";
    auto dataset = geoneric::open_dataset(name, geoneric::OpenMode::READ,
        "AAIGrid");
    BOOST_REQUIRE(dataset);
}

BOOST_AUTO_TEST_SUITE_END()
