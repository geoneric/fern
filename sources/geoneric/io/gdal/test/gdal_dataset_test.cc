#define BOOST_TEST_MODULE geoneric io
#include <boost/test/unit_test.hpp>
#include <gdal_priv.h>
#include "geoneric/core/io_error.h"
#include "geoneric/feature/core/attributes.h"
#include "geoneric/io/gdal/gdal_dataset.h"
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


BOOST_FIXTURE_TEST_SUITE(gdal_dataset, Support)

BOOST_AUTO_TEST_CASE(raster_1)
{
    geoneric::GDALDataset dataset("raster-1.asc", geoneric::OpenMode::READ);
    BOOST_CHECK_EQUAL(dataset.nr_features(), 1u);
    BOOST_CHECK(dataset.contains_feature("/"));
    BOOST_CHECK(dataset.contains_attribute("raster-1"));

    // Read the feature containing the attribute.
    {
        std::shared_ptr<geoneric::Feature> feature = dataset.read_feature("/");
        BOOST_REQUIRE(feature);
        BOOST_CHECK_EQUAL(feature->nr_attributes(), 1u);
        BOOST_CHECK(feature->contains_attribute("raster-1"));

        geoneric::FieldAttributePtr<int32_t> attribute =
            std::dynamic_pointer_cast<geoneric::FieldAttribute<int32_t>>(
                feature->attribute("raster-1"));
        BOOST_REQUIRE(attribute);
        BOOST_REQUIRE_EQUAL(attribute->size(), 1u);

        geoneric::d2::MaskedArrayValue<int32_t> const& value =
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

    // Read the attribute.
    {
        geoneric::FieldAttributePtr<int32_t> attribute =
            std::dynamic_pointer_cast<geoneric::FieldAttribute<int32_t>>(
                dataset.read_attribute("raster-1"));
        BOOST_REQUIRE(attribute);
        BOOST_REQUIRE_EQUAL(attribute->size(), 1u);

        geoneric::d2::MaskedArrayValue<int32_t> const& value =
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
    geoneric::GDALDataset dataset("raster-2.asc", geoneric::OpenMode::READ);
    BOOST_CHECK_EQUAL(dataset.nr_features(), 1u);
    BOOST_CHECK(dataset.contains_feature("/"));
    BOOST_CHECK(dataset.contains_attribute("raster-2"));

    // Read the feature containing the attribute.
    {
        std::shared_ptr<geoneric::Feature> feature = dataset.read_feature("/");
        BOOST_REQUIRE(feature);
        BOOST_CHECK_EQUAL(feature->nr_attributes(), 1u);
        BOOST_CHECK(feature->contains_attribute("raster-2"));

        geoneric::FieldAttributePtr<float> attribute =
            std::dynamic_pointer_cast<geoneric::FieldAttribute<float>>(
                feature->attribute("raster-2"));
        BOOST_REQUIRE(attribute);
        BOOST_REQUIRE_EQUAL(attribute->size(), 1u);

        geoneric::d2::MaskedArrayValue<float> const& value =
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

    try {
        geoneric::GDALDataset dataset(driver, "../does_not_exist.asc",
            geoneric::OpenMode::READ);
        BOOST_CHECK(false);
    }
    catch(geoneric::IOError const& exception) {
        geoneric::String message = exception.message();
        BOOST_CHECK_EQUAL(message,
            "IO error handling ../does_not_exist.asc: Does not exist");
    }

    try {
        geoneric::GDALDataset dataset(driver, "write_only.asc",
            geoneric::OpenMode::READ);
        BOOST_CHECK(false);
    }
    catch(geoneric::IOError const& exception) {
        geoneric::String message = exception.message();
        BOOST_CHECK_EQUAL(message,
            "IO error handling write_only.asc: Cannot be read");
    }

    {
        geoneric::GDALDataset dataset(driver, "raster-1.asc",
            geoneric::OpenMode::READ);

        try {
            dataset.read_feature("blahdiblah");
            BOOST_CHECK(false);
        }
        catch(geoneric::IOError const& exception) {
            geoneric::String message = exception.message();
            BOOST_CHECK_EQUAL(message,
                "IO error handling raster-1.asc: "
                "Does not contain feature: blahdiblah");
        }

        try {
            dataset.read_attribute("blahdiblah");
            BOOST_CHECK(false);
        }
        catch(geoneric::IOError const& exception) {
            geoneric::String message = exception.message();
            BOOST_CHECK_EQUAL(message,
                "IO error handling raster-1.asc: "
                "Does not contain attribute: blahdiblah");
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
