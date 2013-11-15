#define BOOST_TEST_MODULE geoneric io
#include <boost/test/unit_test.hpp>
#include "fern/core/io_error.h"
#include "fern/io/drivers.h"
#include "fern/io/io_client.h"


class Support:
    public fern::IOClient
{

public:

    Support()
        : fern::IOClient()
    {
    }

};


BOOST_FIXTURE_TEST_SUITE(drivers, Support)

BOOST_AUTO_TEST_CASE(drivers)
{
    using namespace fern;

    {
        auto dataset = open_dataset("raster-1.asc", OpenMode::READ, "AAIGrid");
        BOOST_REQUIRE(dataset);
    }

    {
        try {
            open_dataset("../does_not_exist.gnr", OpenMode::READ);
            BOOST_CHECK(false);
        }
        catch(IOError const& exception) {
            String message = exception.message();
            BOOST_CHECK_EQUAL(message,
                "IO error handling ../does_not_exist.gnr: Cannot be read");
        }
    }
}


BOOST_AUTO_TEST_CASE(errors)
{
    using namespace fern;

    for(auto pair: fern::drivers) {
        auto driver = pair.second;

        // When opening for read, the dataset must already exist.
        try {
            driver->open("../does_not_exist.gnr", OpenMode::READ);
            BOOST_CHECK(false);
        }
        catch(IOError const& exception) {
            String message = exception.message();
            BOOST_CHECK_EQUAL(message,
                "IO error handling ../does_not_exist.gnr: Does not exist");
        }

        // When opening for update, the dataset must already exist.
        try {
            driver->open("../does_not_exist.gnr", OpenMode::UPDATE);
            BOOST_CHECK(false);
        }
        catch(IOError const& exception) {
            String message = exception.message();
            BOOST_CHECK_EQUAL(message,
                "IO error handling ../does_not_exist.gnr: Does not exist");
        }

        // When opening for read, the dataset must be readable.
        try {
            driver->open("write_only.gnr", OpenMode::READ);
            BOOST_CHECK(false);
        }
        catch(IOError const& exception) {
            String message = exception.message();
            BOOST_CHECK_EQUAL(message,
                "IO error handling write_only.gnr: Cannot be read");
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
