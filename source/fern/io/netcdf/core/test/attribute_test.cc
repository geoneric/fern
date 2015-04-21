// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern io netcdf core attribute
#include <boost/test/unit_test.hpp>
#include "fern/io/core/file.h"
#include "fern/io/netcdf/core/attribute.h"
#include "fern/io/netcdf/core/dataset.h"


namespace fi = fern::io;
namespace fin = fern::io::netcdf;


BOOST_AUTO_TEST_SUITE(attribute)

BOOST_AUTO_TEST_CASE(has_attribute)
{
    std::string dataset_pathname = "earth.nc";
    BOOST_REQUIRE(fi::file_exists(dataset_pathname));

    fin::DatasetHandle dataset = fin::open_dataset(dataset_pathname);

    BOOST_CHECK(!fin::has_attribute(dataset, "does_not_exist"));
    BOOST_CHECK( fin::has_attribute(dataset, "Conventions"));
}

BOOST_AUTO_TEST_SUITE_END()
