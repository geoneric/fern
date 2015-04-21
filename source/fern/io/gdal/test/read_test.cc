// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern io gdal read
#include <boost/test/unit_test.hpp>
#include "fern/feature/core/data_customization_point/masked_raster.h"
#include "fern/feature/core/data_customization_point/array.h"
#include "fern/algorithm/core/mask_customization_point/array.h"
#include "fern/algorithm/policy/mark_no_data_by_value.h"
#include "fern/io/core/file.h"
#include "fern/io/gdal/gdal_client.h"
#include "fern/io/gdal/read.h"


namespace fa = fern::algorithm;
namespace fi = fern::io;
namespace fig = fern::io::gdal;


using Raster = fern::MaskedRaster<int32_t, 2>;
using Mask = Raster::Mask;
using OutputNoDataPolicy = fa::MarkNoDataByValue<Mask>;


BOOST_FIXTURE_TEST_SUITE(read_, fig::GDALClient)

BOOST_AUTO_TEST_CASE(file_does_no_exist)
{
    bool exception_thrown{false};

    try {
        fern::DataName data_name{"does_not_exist.tif:/does_not_exist"};
        size_t const nr_rows{3};
        size_t const nr_cols{2};
        Raster::Transformation transformation{{0.0, 1.0, 0.0, 1.0}};
        Raster raster(fern::extents[nr_rows][nr_cols], transformation);
        OutputNoDataPolicy output_no_data_policy(raster.mask());

        BOOST_REQUIRE(!fi::file_exists("does_not_exist.tif"));
        fig::read(output_no_data_policy, data_name, raster);
    }
    catch(fern::IOError const& exception) {
        std::string message = exception.message();
        BOOST_CHECK_EQUAL(message,
                "I/O error handling does_not_exist.tif: "
                "Does not exist");
        exception_thrown = true;
    }

    BOOST_CHECK(exception_thrown);
}


BOOST_AUTO_TEST_CASE(is_not_readable)
{
    bool exception_thrown{false};

    try {
        fern::DataName data_name{"unreadable.tif:/unreadable"};
        size_t const nr_rows{3};
        size_t const nr_cols{2};
        Raster::Transformation transformation{{0.0, 1.0, 0.0, 1.0}};
        Raster raster(fern::extents[nr_rows][nr_cols], transformation);
        OutputNoDataPolicy output_no_data_policy(raster.mask());

        BOOST_REQUIRE(fi::file_exists("unreadable.tif"));
        fig::read(output_no_data_policy, data_name, raster);
    }
    catch(fern::IOError const& exception) {
        std::string message = exception.message();
        BOOST_CHECK_EQUAL(message,
                "I/O error handling unreadable.tif: "
                "Cannot be read");
        exception_thrown = true;
    }

    BOOST_CHECK(exception_thrown);
}


BOOST_AUTO_TEST_CASE(raster)
{
    fern::DataName data_name{"soil.tif:/soil"};
    size_t const nr_rows{3};
    size_t const nr_cols{2};
    Raster::Transformation transformation{{30.0, 2.5, 40.0, 2.5}};
    Raster raster(fern::extents[nr_rows][nr_cols], transformation);
    OutputNoDataPolicy output_no_data_policy(raster.mask());

    BOOST_REQUIRE(fi::file_exists("soil.tif"));
    fig::read(output_no_data_policy, data_name, raster);

    BOOST_CHECK_CLOSE(raster.transformation()[0], 30.0, 7);
    BOOST_CHECK_CLOSE(raster.transformation()[1], 2.5, 7);
    BOOST_CHECK_CLOSE(raster.transformation()[2], 40.0, 7);
    BOOST_CHECK_CLOSE(raster.transformation()[3], 2.5, 7);

    BOOST_CHECK(!raster.mask().data()[0]);
    BOOST_CHECK(!raster.mask().data()[1]);
    BOOST_CHECK(!raster.mask().data()[2]);
    BOOST_CHECK( raster.mask().data()[3]);
    BOOST_CHECK(!raster.mask().data()[4]);
    BOOST_CHECK(!raster.mask().data()[5]);

    BOOST_CHECK_EQUAL(raster.data()[0], -2);
    BOOST_CHECK_EQUAL(raster.data()[1], -1);
    BOOST_CHECK_EQUAL(raster.data()[2], 0);
    BOOST_CHECK_EQUAL(raster.data()[4], 1);
    BOOST_CHECK_EQUAL(raster.data()[5], 2);
}

BOOST_AUTO_TEST_SUITE_END()
