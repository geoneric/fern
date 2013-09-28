#define BOOST_TEST_MODULE geoneric operation_core
#include <boost/test/unit_test.hpp>
#include "geoneric/core/data_types.h"


BOOST_AUTO_TEST_SUITE(data_types)

BOOST_AUTO_TEST_CASE(string)
{
    BOOST_CHECK_EQUAL(geoneric::DataTypes::UNKNOWN.to_string(), "?");
    // BOOST_CHECK_EQUAL(geoneric::DataTypes::FEATURE.to_string(),
    //     "Point|Line|Polygon");
    BOOST_CHECK_EQUAL(geoneric::DataTypes::ALL.to_string(),
        "Constant|StaticField");
    // BOOST_CHECK_EQUAL(
    //     (geoneric::DataTypes::CONSTANT | geoneric::DataTypes::FEATURE).to_string(),
    //     "Constant|Point|Line|Polygon");
}


BOOST_AUTO_TEST_CASE(unknown)
{
    geoneric::DataTypes data_types;

    BOOST_CHECK_EQUAL(data_types, geoneric::DataTypes::UNKNOWN);
}


BOOST_AUTO_TEST_CASE(add)
{
    geoneric::DataTypes data_types;
    // data_types |= geoneric::DataTypes::POINT;
    // data_types |= geoneric::DataTypes::LINE | geoneric::DataTypes::POLYGON;

    // BOOST_CHECK_EQUAL(data_types, geoneric::DataTypes::FEATURE);

    data_types |= geoneric::DataTypes::CONSTANT;
    data_types |= geoneric::DataTypes::STATIC_FIELD;
    BOOST_CHECK_EQUAL(data_types, geoneric::DataTypes::ALL);

    data_types = geoneric::DataTypes::CONSTANT |
        geoneric::DataTypes::STATIC_FIELD;
    BOOST_CHECK_EQUAL(data_types, geoneric::DataTypes::ALL);
}


BOOST_AUTO_TEST_CASE(remove)
{
    geoneric::DataTypes data_types;

    data_types = geoneric::DataTypes::CONSTANT |
        geoneric::DataTypes::STATIC_FIELD;
    data_types ^= geoneric::DataTypes::CONSTANT;
    BOOST_CHECK_EQUAL(data_types, geoneric::DataTypes::STATIC_FIELD);

    data_types = geoneric::DataTypes::CONSTANT |
        geoneric::DataTypes::STATIC_FIELD;

    data_types ^= geoneric::DataTypes::CONSTANT |
        geoneric::DataTypes::STATIC_FIELD;
    BOOST_CHECK_EQUAL(data_types, geoneric::DataTypes::UNKNOWN);
}


BOOST_AUTO_TEST_CASE(count)
{
    geoneric::DataTypes data_types;

    BOOST_CHECK_EQUAL(geoneric::DataTypes::UNKNOWN.count(), 0u);
    BOOST_CHECK_EQUAL(geoneric::DataTypes::CONSTANT.count(), 1u);
    BOOST_CHECK_EQUAL(geoneric::DataTypes::ALL.count(), 2u);
}

BOOST_AUTO_TEST_SUITE_END()
