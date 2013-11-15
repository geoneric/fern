#define BOOST_TEST_MODULE fern feature
#include <boost/test/unit_test.hpp>
#include "fern/feature/core/attributes.h"


BOOST_AUTO_TEST_SUITE(feature)

BOOST_AUTO_TEST_CASE(feature)
{
    // Earth feature. All attributes are relevant for this one planet.
    {
        fern::Feature earth;
        BOOST_CHECK_EQUAL(earth.nr_features(), 0u);
        BOOST_CHECK(!earth.contains_feature("continents"));
        BOOST_CHECK_EQUAL(earth.nr_attributes(), 0u);
        BOOST_CHECK(!earth.contains_attribute("acceleration"));

        earth.add_attribute("acceleration",
            std::make_shared<fern::ConstantAttribute<double>>(9.80665));
        BOOST_CHECK_EQUAL(earth.nr_features(), 0u);
        BOOST_CHECK_EQUAL(earth.nr_attributes(), 1u);
        BOOST_CHECK(earth.contains_attribute("acceleration"));

        // Add continents child-feature.
        earth.add_feature("continents", std::make_shared<fern::Feature>());
        BOOST_CHECK_EQUAL(earth.nr_features(), 1u);
        BOOST_CHECK(earth.contains_feature("continents"));
        BOOST_CHECK_EQUAL(earth.nr_attributes(), 1u);

        // Add attribute that is global to all continents.
        earth.add_attribute("/continents/is_land",
            std::make_shared<fern::ConstantAttribute<bool>>(true));
        BOOST_CHECK_EQUAL(earth.nr_features(), 1u);
        BOOST_CHECK_EQUAL(earth.nr_attributes(), 1u);
        BOOST_CHECK_EQUAL(earth.nr_features("continents"), 0u);
        BOOST_CHECK_EQUAL(earth.nr_attributes("continents"), 1u);
        BOOST_CHECK(earth.contains_attribute("continents/is_land"));
    }

    // Planets feature. Attributes are stored per planet (a point in space).
    {
        fern::Feature planets;

        typedef int Value;
        typedef fern::Point<int, 3> Point;
        typedef fern::SpatialAttribute<fern::SpatialDomain<Point>,
            Value> PointsAttribute;
        typedef std::shared_ptr<PointsAttribute> PointsAttributePtr;

        Point center_of_planet1 = { 1, 1, 1 };
        PointsAttributePtr acceleration(new PointsAttribute());
        acceleration->add(center_of_planet1, 1.1);
        planets.add_attribute("acceleration", acceleration);
        BOOST_CHECK_EQUAL(planets.nr_attributes(), 1);
    }
}

BOOST_AUTO_TEST_SUITE_END()
