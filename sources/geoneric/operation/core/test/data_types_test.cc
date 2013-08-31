#define BOOST_TEST_MODULE ranally operation_core
#include <boost/test/unit_test.hpp>
#include "ranally/operation/core/data_types.h"


BOOST_AUTO_TEST_SUITE(data_types)

BOOST_AUTO_TEST_CASE(string)
{
    BOOST_CHECK_EQUAL(ranally::DataTypes::UNKNOWN.to_string(), "?");
    BOOST_CHECK_EQUAL(ranally::DataTypes::FEATURE.to_string(),
        "Point|Line|Polygon");
    BOOST_CHECK_EQUAL(ranally::DataTypes::ALL.to_string(),
        "Scalar|Point|Line|Polygon");
    BOOST_CHECK_EQUAL(
        (ranally::DataTypes::SCALAR | ranally::DataTypes::FEATURE).to_string(),
        "Scalar|Point|Line|Polygon");
}


BOOST_AUTO_TEST_CASE(unknown)
{
    ranally::DataTypes data_types;

    BOOST_CHECK_EQUAL(data_types, ranally::DataTypes::UNKNOWN);
}


BOOST_AUTO_TEST_CASE(add)
{
    ranally::DataTypes data_types;
    data_types |= ranally::DataTypes::POINT;
    data_types |= ranally::DataTypes::LINE | ranally::DataTypes::POLYGON;

    BOOST_CHECK_EQUAL(data_types, ranally::DataTypes::FEATURE);
}


BOOST_AUTO_TEST_CASE(remove)
{
    ranally::DataTypes data_types;

    data_types = ranally::DataTypes::FEATURE;
    data_types ^= ranally::DataTypes::POINT;

    BOOST_CHECK_EQUAL(data_types,
        ranally::DataTypes::LINE | ranally::DataTypes::POLYGON);

    data_types = ranally::DataTypes::FEATURE;
    data_types ^= ranally::DataTypes::POINT | ranally::DataTypes::POLYGON;
    BOOST_CHECK_EQUAL(data_types, ranally::DataTypes::LINE);
}


BOOST_AUTO_TEST_CASE(count)
{
    ranally::DataTypes data_types;

    BOOST_CHECK_EQUAL(ranally::DataTypes::UNKNOWN.count(), 0u);
    BOOST_CHECK_EQUAL(ranally::DataTypes::POINT.count(), 1u);
    BOOST_CHECK_EQUAL(ranally::DataTypes::LINE.count(), 1u);
    BOOST_CHECK_EQUAL(ranally::DataTypes::FEATURE.count(), 3u);
    // BOOST_CHECK_EQUAL(ranally::DataTypes::DEPENDS_ON_INPUT.count(), 1u);
}

BOOST_AUTO_TEST_SUITE_END()
