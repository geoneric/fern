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
#include "fern/core/io_error.h"
#include "fern/language/io/drivers.h"
#include "fern/language/io/io_client.h"


namespace fl = fern::language;


BOOST_FIXTURE_TEST_SUITE(drivers, fl::IOClient)

BOOST_AUTO_TEST_CASE(drivers)
{
    {
        auto dataset = fl::open_dataset("raster-1.asc", fl::OpenMode::READ,
            "AAIGrid");
        BOOST_REQUIRE(dataset);
    }

    {
        try {
            fl::open_dataset("../does_not_exist.h5", fl::OpenMode::READ);
            BOOST_CHECK(false);
        }
        catch(fern::IOError const& exception) {
            std::string message = exception.message();
            BOOST_CHECK_EQUAL(message,
                "I/O error handling ../does_not_exist.h5: Cannot be read");
        }
    }
}


BOOST_AUTO_TEST_CASE(errors)
{
    for(auto pair: fl::drivers) {
        auto driver = pair.second;

        // When opening for read, the dataset must already exist.
        try {
            driver->open("../does_not_exist.h5", fl::OpenMode::READ);
            BOOST_CHECK(false);
        }
        catch(fern::IOError const& exception) {
            std::string message = exception.message();
            BOOST_CHECK_EQUAL(message,
                "I/O error handling ../does_not_exist.h5: Does not exist");
        }

        // When opening for update, the dataset must already exist.
        try {
            driver->open("../does_not_exist.h5", fl::OpenMode::UPDATE);
            BOOST_CHECK(false);
        }
        catch(fern::IOError const& exception) {
            std::string message = exception.message();
            BOOST_CHECK_EQUAL(message,
                "I/O error handling ../does_not_exist.h5: Does not exist");
        }

        // When opening for read, the dataset must be readable.
        try {
            driver->open("write_only.h5", fl::OpenMode::READ);
            BOOST_CHECK(false);
        }
        catch(fern::IOError const& exception) {
            std::string message = exception.message();
            BOOST_CHECK_EQUAL(message,
                "I/O error handling write_only.h5: Cannot be read");
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
