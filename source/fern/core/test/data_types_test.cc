// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern core data_types
#include <boost/test/unit_test.hpp>
#include "fern/core/data_types.h"


BOOST_AUTO_TEST_CASE(string)
{
    BOOST_CHECK_EQUAL(fern::DataTypes::UNKNOWN.to_string(), "?");
    // BOOST_CHECK_EQUAL(fern::DataTypes::FEATURE.to_string(),
    //     "Point|Line|Polygon");
    BOOST_CHECK_EQUAL(fern::DataTypes::ALL.to_string(),
        "Constant|StaticField");
    // BOOST_CHECK_EQUAL(
    //     (fern::DataTypes::CONSTANT | fern::DataTypes::FEATURE).to_string(),
    //     "Constant|Point|Line|Polygon");
}


BOOST_AUTO_TEST_CASE(unknown)
{
    fern::DataTypes data_types;

    BOOST_CHECK_EQUAL(data_types, fern::DataTypes::UNKNOWN);
}


BOOST_AUTO_TEST_CASE(add)
{
    fern::DataTypes data_types;
    // data_types |= fern::DataTypes::POINT;
    // data_types |= fern::DataTypes::LINE | fern::DataTypes::POLYGON;

    // BOOST_CHECK_EQUAL(data_types, fern::DataTypes::FEATURE);

    data_types |= fern::DataTypes::CONSTANT;
    data_types |= fern::DataTypes::STATIC_FIELD;
    BOOST_CHECK_EQUAL(data_types, fern::DataTypes::ALL);

    data_types = fern::DataTypes::CONSTANT |
        fern::DataTypes::STATIC_FIELD;
    BOOST_CHECK_EQUAL(data_types, fern::DataTypes::ALL);
}


BOOST_AUTO_TEST_CASE(remove_)
{
    fern::DataTypes data_types;

    data_types = fern::DataTypes::CONSTANT |
        fern::DataTypes::STATIC_FIELD;
    data_types ^= fern::DataTypes::CONSTANT;
    BOOST_CHECK_EQUAL(data_types, fern::DataTypes::STATIC_FIELD);

    data_types = fern::DataTypes::CONSTANT |
        fern::DataTypes::STATIC_FIELD;

    data_types ^= fern::DataTypes::CONSTANT |
        fern::DataTypes::STATIC_FIELD;
    BOOST_CHECK_EQUAL(data_types, fern::DataTypes::UNKNOWN);
}


BOOST_AUTO_TEST_CASE(count)
{
    fern::DataTypes data_types;

    BOOST_CHECK_EQUAL(fern::DataTypes::UNKNOWN.count(), 0u);
    BOOST_CHECK_EQUAL(fern::DataTypes::CONSTANT.count(), 1u);
    BOOST_CHECK_EQUAL(fern::DataTypes::ALL.count(), 2u);
}
