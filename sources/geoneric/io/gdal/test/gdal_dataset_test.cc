#define BOOST_TEST_MODULE geoneric io
#include <boost/test/unit_test.hpp>
#include "gdal_priv.h"
#include "geoneric/feature/core/array_value.h"
#include "geoneric/feature/core/box.h"
#include "geoneric/feature/core/point.h"
#include "geoneric/feature/core/spatial_attribute.h"
#include "geoneric/feature/core/spatial_domain.h"
#include "geoneric/io/gdal/gdal_dataset.h"


BOOST_AUTO_TEST_SUITE(gdal_dataset)

BOOST_AUTO_TEST_CASE(gdal_dataset)
{
    GDALAllRegister();

    geoneric::GDALDataset dataset("raster-1.asc");
    BOOST_CHECK_EQUAL(dataset.nr_features(), 1u);
    BOOST_CHECK(dataset.contains_feature("raster-1"));

    std::shared_ptr<geoneric::Feature> feature = dataset.read("raster-1");
    BOOST_CHECK_EQUAL(feature->nr_attributes(), 1u);
    BOOST_CHECK(feature->contains_attribute("raster-1"));

    typedef geoneric::Point<double, 2> Point;
    typedef geoneric::Box<Point> Box;
    typedef geoneric::SpatialDomain<Box> BoxDomain;
    typedef geoneric::ArrayValue<int32_t, 1> Value;
    typedef std::shared_ptr<Value> ValuePtr;
    typedef geoneric::SpatialAttribute<BoxDomain, ValuePtr> BoxesAttribute;
    typedef std::shared_ptr<BoxesAttribute> BoxesAttributePtr;

    BoxesAttributePtr attribute = std::dynamic_pointer_cast<BoxesAttribute>(
        feature->attribute("raster-1"));
    BOOST_REQUIRE(attribute);
    BOOST_REQUIRE_EQUAL(attribute->size(), 1u);

    ValuePtr value = attribute->values().cbegin()->second;
    BOOST_REQUIRE_EQUAL(value->num_dimensions(), 1);
    BOOST_REQUIRE_EQUAL(value->size(), 6);

    BOOST_CHECK_EQUAL((*value)[0],    -2);
    BOOST_CHECK_EQUAL((*value)[1],    -1);
    BOOST_CHECK_EQUAL((*value)[2],    -0);
    BOOST_CHECK_EQUAL((*value)[3], -9999);
    BOOST_CHECK_EQUAL((*value)[4],     1);
    BOOST_CHECK_EQUAL((*value)[5],     2);
}

BOOST_AUTO_TEST_SUITE_END()
