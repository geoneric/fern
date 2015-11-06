// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern io netcdf core variable
#include <boost/test/unit_test.hpp>
#include "fern/algorithm/policy/dont_mark_no_data.h"
#include "fern/core/data_customization_point/scalar.h"
#include "fern/io/core/file.h"
#include "fern/io/netcdf/core/dataset.h"
#include "fern/io/netcdf/core/variable.h"


namespace fa = fern::algorithm;
namespace fi = fern::io;
namespace fin = fern::io::netcdf;


BOOST_AUTO_TEST_CASE(contains_variable)
{
    std::string dataset_pathname = "earth.nc";
    BOOST_REQUIRE(fi::file_exists(dataset_pathname));

    fin::DatasetHandle dataset = fin::open_dataset(dataset_pathname);

    BOOST_CHECK(!fin::contains_variable(dataset, "does_not_exist"));
    BOOST_CHECK( fin::contains_variable(dataset, "gravity"));
}


BOOST_AUTO_TEST_CASE(variable_is_scalar)
{
    std::string dataset_pathname = "earth.nc";
    BOOST_REQUIRE(fi::file_exists(dataset_pathname));

    fin::DatasetHandle dataset = fin::open_dataset(dataset_pathname);
    std::string variable_name = "gravity";
    BOOST_REQUIRE(fin::contains_variable(dataset, variable_name));

    int variable_id = fin::variable_id(dataset, variable_name);
    BOOST_CHECK(fin::variable_is_scalar(dataset, variable_id));

    // TODO Test non-scalar variable.
}


BOOST_AUTO_TEST_CASE(value_type_id)
{
    std::string dataset_pathname = "earth.nc";
    fin::DatasetHandle dataset = fin::open_dataset(dataset_pathname);
    std::string variable_name = "gravity";
    int variable_id = fin::variable_id(dataset, variable_name);
    BOOST_CHECK_EQUAL(fin::value_type_id(dataset, variable_id),
        fern::VT_FLOAT64);
}


BOOST_AUTO_TEST_CASE(read_variable)
{
    std::string dataset_pathname = "earth.nc";
    fin::DatasetHandle dataset = fin::open_dataset(dataset_pathname);
    std::string variable_name = "gravity";
    int variable_id = fin::variable_id(dataset, variable_name);
    double gravity;
    fa::DontMarkNoData output_no_data_policy;
    fin::read_variable(output_no_data_policy, dataset, variable_id, gravity);

    BOOST_CHECK_CLOSE(gravity, 9.8, 7);
}
