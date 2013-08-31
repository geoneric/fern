#define BOOST_TEST_MODULE geoneric operation_core
#include <boost/test/unit_test.hpp>
#include "geoneric/operation/core/data_types.h"


BOOST_AUTO_TEST_SUITE(data_types)

BOOST_AUTO_TEST_CASE(string)
{
    BOOST_CHECK_EQUAL(geoneric::DataTypes::UNKNOWN.to_string(), "?");
    BOOST_CHECK_EQUAL(geoneric::DataTypes::FEATURE.to_string(),
        "Point|Line|Polygon");
    BOOST_CHECK_EQUAL(geoneric::DataTypes::ALL.to_string(),
        "Scalar|Point|Line|Polygon");
    BOOST_CHECK_EQUAL(
        (geoneric::DataTypes::SCALAR | geoneric::DataTypes::FEATURE).to_string(),
        "Scalar|Point|Line|Polygon");
}


BOOST_AUTO_TEST_CASE(unknown)
{
    geoneric::DataTypes data_types;

    BOOST_CHECK_EQUAL(data_types, geoneric::DataTypes::UNKNOWN);
}


BOOST_AUTO_TEST_CASE(add)
{
    geoneric::DataTypes data_types;
    data_types |= geoneric::DataTypes::POINT;
    data_types |= geoneric::DataTypes::LINE | geoneric::DataTypes::POLYGON;

    BOOST_CHECK_EQUAL(data_types, geoneric::DataTypes::FEATURE);
}


BOOST_AUTO_TEST_CASE(remove)
{
    geoneric::DataTypes data_types;

    data_types = geoneric::DataTypes::FEATURE;
    data_types ^= geoneric::DataTypes::POINT;

    BOOST_CHECK_EQUAL(data_types,
        geoneric::DataTypes::LINE | geoneric::DataTypes::POLYGON);

    data_types = geoneric::DataTypes::FEATURE;
    data_types ^= geoneric::DataTypes::POINT | geoneric::DataTypes::POLYGON;
    BOOST_CHECK_EQUAL(data_types, geoneric::DataTypes::LINE);
}


BOOST_AUTO_TEST_CASE(count)
{
    geoneric::DataTypes data_types;

    BOOST_CHECK_EQUAL(geoneric::DataTypes::UNKNOWN.count(), 0u);
    BOOST_CHECK_EQUAL(geoneric::DataTypes::POINT.count(), 1u);
    BOOST_CHECK_EQUAL(geoneric::DataTypes::LINE.count(), 1u);
    BOOST_CHECK_EQUAL(geoneric::DataTypes::FEATURE.count(), 3u);
    // BOOST_CHECK_EQUAL(geoneric::DataTypes::DEPENDS_ON_INPUT.count(), 1u);
}

BOOST_AUTO_TEST_SUITE_END()
