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
#include <gdal_priv.h>
#include "fern/core/io_error.h"
#include "fern/language/feature/core/attributes.h"
#include "fern/language/io/gdal/gdal_dataset.h"


namespace fl = fern::language;


class Support
{

public:

    Support()
    {
        GDALAllRegister();
    }

};


BOOST_FIXTURE_TEST_SUITE(gdal_dataset, Support)

BOOST_AUTO_TEST_CASE(raster_1)
{
    fl::GDALDataset dataset("AAIGRID", "raster-1.asc", fl::OpenMode::READ);
    BOOST_CHECK_EQUAL(dataset.nr_features(), 1u);
    BOOST_CHECK(dataset.contains_feature("/raster-1"));
    BOOST_CHECK(dataset.contains_attribute("/raster-1/raster-1"));

    auto feature_names = dataset.feature_names();
    BOOST_REQUIRE_EQUAL(feature_names.size(), 1u);
    BOOST_CHECK_EQUAL(feature_names[0], "raster-1");

    // Read the feature containing the attribute.
    {
        std::shared_ptr<fl::Feature> feature = dataset.read_feature(
            "/raster-1");
        BOOST_REQUIRE(feature);
        BOOST_CHECK_EQUAL(feature->nr_attributes(), 1u);
        BOOST_CHECK(feature->contains_attribute("raster-1"));

        fl::FieldAttributePtr<int32_t> attribute =
            std::dynamic_pointer_cast<fl::FieldAttribute<int32_t>>(
                feature->attribute("raster-1"));
        BOOST_REQUIRE(attribute);
        BOOST_REQUIRE_EQUAL(attribute->size(), 1u);

        fl::d2::MaskedArrayValue<int32_t> const& value =
            *attribute->values().cbegin()->second;
        BOOST_REQUIRE_EQUAL(value.num_dimensions(), 2);
        BOOST_REQUIRE_EQUAL(value.shape()[0], 3);
        BOOST_REQUIRE_EQUAL(value.shape()[1], 2);

        BOOST_CHECK(!value.mask()[0][0]);
        BOOST_CHECK(!value.mask()[0][1]);
        BOOST_CHECK(!value.mask()[1][0]);
        BOOST_CHECK( value.mask()[1][1]);
        BOOST_CHECK(!value.mask()[2][0]);
        BOOST_CHECK(!value.mask()[2][1]);
        BOOST_CHECK_EQUAL(value[0][0], -2);
        BOOST_CHECK_EQUAL(value[0][1], -1);
        BOOST_CHECK_EQUAL(value[1][0], -0);
        // BOOST_CHECK_EQUAL(value[1][1], -9999);
        BOOST_CHECK_EQUAL(value[2][0], 1);
        BOOST_CHECK_EQUAL(value[2][1], 2);
    }

    // Open the attribute.
    {
        fl::FieldAttributePtr<int32_t> attribute =
            std::dynamic_pointer_cast<fl::FieldAttribute<int32_t>>(
                dataset.open_attribute("/raster-1/raster-1"));
        BOOST_REQUIRE(attribute);
        BOOST_REQUIRE_EQUAL(attribute->size(), 0u);
    }

    // Read the attribute.
    {
        fl::FieldAttributePtr<int32_t> attribute =
            std::dynamic_pointer_cast<fl::FieldAttribute<int32_t>>(
                dataset.read_attribute("/raster-1/raster-1"));
        BOOST_REQUIRE(attribute);
        BOOST_REQUIRE_EQUAL(attribute->size(), 1u);

        fl::d2::MaskedArrayValue<int32_t> const& value =
            *attribute->values().cbegin()->second;
        BOOST_REQUIRE_EQUAL(value.num_dimensions(), 2);
        BOOST_REQUIRE_EQUAL(value.shape()[0], 3);
        BOOST_REQUIRE_EQUAL(value.shape()[1], 2);

        BOOST_CHECK(!value.mask()[0][0]);
        BOOST_CHECK(!value.mask()[0][1]);
        BOOST_CHECK(!value.mask()[1][0]);
        BOOST_CHECK( value.mask()[1][1]);
        BOOST_CHECK(!value.mask()[2][0]);
        BOOST_CHECK(!value.mask()[2][1]);
        BOOST_CHECK_EQUAL(value[0][0], -2);
        BOOST_CHECK_EQUAL(value[0][1], -1);
        BOOST_CHECK_EQUAL(value[1][0], -0);
        // BOOST_CHECK_EQUAL(value[1][1], -9999);
        BOOST_CHECK_EQUAL(value[2][0], 1);
        BOOST_CHECK_EQUAL(value[2][1], 2);
    }
}


BOOST_AUTO_TEST_CASE(raster_2)
{
    fl::GDALDataset dataset("AAIGRID", "raster-2.asc", fl::OpenMode::READ);
    BOOST_CHECK_EQUAL(dataset.nr_features(), 1u);
    BOOST_CHECK(dataset.contains_feature("/raster-2"));
    BOOST_CHECK(dataset.contains_attribute("/raster-2/raster-2"));

    // Read the feature containing the attribute.
    {
        std::shared_ptr<fl::Feature> feature = dataset.read_feature(
            "/raster-2");
        BOOST_REQUIRE(feature);
        BOOST_CHECK_EQUAL(feature->nr_attributes(), 1u);
        BOOST_CHECK(feature->contains_attribute("raster-2"));

        fl::FieldAttributePtr<float> attribute =
            std::dynamic_pointer_cast<fl::FieldAttribute<float>>(
                feature->attribute("raster-2"));
        BOOST_REQUIRE(attribute);
        BOOST_REQUIRE_EQUAL(attribute->size(), 1u);

        fl::d2::MaskedArrayValue<float> const& value =
            *attribute->values().cbegin()->second;
        BOOST_REQUIRE_EQUAL(value.num_dimensions(), 2);
        BOOST_REQUIRE_EQUAL(value.shape()[0], 3);
        BOOST_REQUIRE_EQUAL(value.shape()[1], 2);

        BOOST_CHECK(!value.mask()[0][0]);
        BOOST_CHECK(!value.mask()[0][1]);
        BOOST_CHECK(!value.mask()[1][0]);
        BOOST_CHECK( value.mask()[1][1]);
        BOOST_CHECK(!value.mask()[2][0]);
        BOOST_CHECK(!value.mask()[2][1]);
        BOOST_CHECK_CLOSE(value[0][0], -2.2, 0.0001);
        BOOST_CHECK_CLOSE(value[0][1], -1.1, 0.0001);
        BOOST_CHECK_CLOSE(value[1][0], -0.0, 0.0001);
        BOOST_CHECK_CLOSE(value[2][0], 1.1, 0.0001);
        BOOST_CHECK_CLOSE(value[2][1], 2.2, 0.0001);
    }
}


BOOST_AUTO_TEST_CASE(errors)
{
    ::GDALDriver* driver = GetGDALDriverManager()->GetDriverByName("AAIGrid");

    {
        fl::GDALDataset dataset(driver, "raster-1.asc", fl::OpenMode::READ);

        try {
            dataset.read_feature("blahdiblah");
            BOOST_CHECK(false);
        }
        catch(fern::IOError const& exception) {
            std::string message = exception.message();
            BOOST_CHECK_EQUAL(message,
                "I/O error handling raster-1.asc: "
                "Does not contain feature: blahdiblah");
        }

        try {
            dataset.read_attribute("blahdiblah");
            BOOST_CHECK(false);
        }
        catch(fern::IOError const& exception) {
            std::string message = exception.message();
            BOOST_CHECK_EQUAL(message,
                "I/O error handling raster-1.asc: "
                "Does not contain attribute: blahdiblah");
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
