#define BOOST_TEST_MODULE geoneric io
#include <boost/test/unit_test.hpp>
#include "gdal_priv.h"
#include "geoneric/feature/core/attributes.h"
#include "geoneric/io/gdal/gdal_dataset.h"


BOOST_AUTO_TEST_SUITE(gdal_dataset)

BOOST_AUTO_TEST_CASE(gdal_dataset)
{
    GDALAllRegister();

    geoneric::GDALDataset dataset("raster-1.asc");
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
        BOOST_REQUIRE_EQUAL(value.num_elements(), 6);
        BOOST_REQUIRE_EQUAL(value.size(), 3);
        BOOST_REQUIRE_EQUAL(value[0].size(), 2);

        BOOST_CHECK(!value.mask()[0][0]);
        BOOST_CHECK(!value.mask()[0][1]);
        BOOST_CHECK(!value.mask()[1][0]);
        BOOST_CHECK( value.mask()[1][1]);
        BOOST_CHECK(!value.mask()[2][0]);
        BOOST_CHECK(!value.mask()[2][1]);
        BOOST_CHECK_EQUAL(value[0][0],    -2);
        BOOST_CHECK_EQUAL(value[0][1],    -1);
        BOOST_CHECK_EQUAL(value[1][0],    -0);
        // BOOST_CHECK_EQUAL(value[1][1], -9999);
        BOOST_CHECK_EQUAL(value[2][0],     1);
        BOOST_CHECK_EQUAL(value[2][1],     2);
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
        BOOST_REQUIRE_EQUAL(value.num_elements(), 6);
        BOOST_REQUIRE_EQUAL(value.size(), 3);
        BOOST_REQUIRE_EQUAL(value[0].size(), 2);

        BOOST_CHECK(!value.mask()[0][0]);
        BOOST_CHECK(!value.mask()[0][1]);
        BOOST_CHECK(!value.mask()[1][0]);
        BOOST_CHECK( value.mask()[1][1]);
        BOOST_CHECK(!value.mask()[2][0]);
        BOOST_CHECK(!value.mask()[2][1]);
        BOOST_CHECK_EQUAL(value[0][0],    -2);
        BOOST_CHECK_EQUAL(value[0][1],    -1);
        BOOST_CHECK_EQUAL(value[1][0],    -0);
        // BOOST_CHECK_EQUAL(value[1][1], -9999);
        BOOST_CHECK_EQUAL(value[2][0],     1);
        BOOST_CHECK_EQUAL(value[2][1],     2);
    }
}

BOOST_AUTO_TEST_SUITE_END()
