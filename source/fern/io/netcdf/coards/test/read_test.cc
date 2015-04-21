// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern io netcdf coards read
#include <boost/test/unit_test.hpp>
#include "fern/core/data_customization_point/scalar.h"
#include "fern/algorithm/policy/dont_mark_no_data.h"
#include "fern/io/netcdf/coards/read.h"


namespace fa = fern::algorithm;
namespace fi = fern::io;
namespace fin = fern::io::netcdf;


BOOST_AUTO_TEST_SUITE(read_)

BOOST_AUTO_TEST_CASE(file_does_no_exist)
{
    bool exception_thrown{false};

    try {
        fa::DontMarkNoData output_no_data_policy;
        fern::DataName data_name{"does_not_exist.nc:/age"};
        double age;
        fin::read_coards(output_no_data_policy, data_name, age);
    }
    catch(fern::IOError const& exception) {
        std::string message = exception.message();
        BOOST_CHECK_EQUAL(message,
                "I/O error handling does_not_exist.nc: "
                "Does not exist");
        exception_thrown = true;
    }

    BOOST_CHECK(exception_thrown);
}


BOOST_AUTO_TEST_CASE(is_not_readable)
{
    bool exception_thrown{false};

    try {
        fa::DontMarkNoData output_no_data_policy;
        fern::DataName data_name{"unreadable.nc:/age"};
        BOOST_CHECK(fi::file_exists("unreadable.nc"));
        double age;
        fin::read_coards(output_no_data_policy, data_name, age);
    }
    catch(fern::IOError const& exception) {
        std::string message = exception.message();
        BOOST_CHECK_EQUAL(message,
                "I/O error handling unreadable.nc: "
                "Cannot be read");
        exception_thrown = true;
    }

    BOOST_CHECK(exception_thrown);
}


BOOST_AUTO_TEST_CASE(invalid_netcdf)
{
    bool exception_thrown{false};

    try {
        fa::DontMarkNoData output_no_data_policy;
        fern::DataName data_name{"invalid_netcdf.nc:/age"};
        BOOST_CHECK(fi::file_exists("invalid_netcdf.nc"));
        double age;
        fin::read_coards(output_no_data_policy, data_name, age);
    }
    catch(fern::IOError const& exception) {
        std::string message = exception.message();
        BOOST_CHECK_EQUAL(message,
                "I/O error handling invalid_netcdf.nc: "
                "Cannot be read");
        exception_thrown = true;
    }

    BOOST_CHECK(exception_thrown);
}


BOOST_AUTO_TEST_CASE(invalid_netcdf_coards)
{
    bool exception_thrown{false};

    try {
        fa::DontMarkNoData output_no_data_policy;
        fern::DataName data_name{"invalid_netcdf_coards.nc:/age"};
        BOOST_CHECK(fi::file_exists("invalid_netcdf_coards.nc"));
        double age;
        fin::read_coards(output_no_data_policy, data_name, age);
    }
    catch(fern::IOError const& exception) {
        std::string message = exception.message();
        BOOST_CHECK_EQUAL(message,
                "I/O error handling invalid_netcdf_coards.nc: "
                "Does not conform to convention: COARDS");
        exception_thrown = true;
    }

    BOOST_CHECK(exception_thrown);
}


BOOST_AUTO_TEST_CASE(scalar)
{
    fa::DontMarkNoData output_no_data_policy;

    fern::DataName data_name{"earth.nc:/age"};

    std::string pathname{data_name.database_pathname().native_string()};
    BOOST_REQUIRE(fi::file_exists(pathname));

    double age;
    fin::read_coards(output_no_data_policy, data_name, age);
    BOOST_CHECK_CLOSE(age, 4.54e9, 7);
}

BOOST_AUTO_TEST_SUITE_END()
