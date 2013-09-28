#define BOOST_TEST_MODULE geoneric feature
#include <boost/test/unit_test.hpp>
#include "geoneric/feature/core/attributes.h"
// #include "geoneric/feature/core/feature.h"


BOOST_AUTO_TEST_SUITE(feature)

BOOST_AUTO_TEST_CASE(feature)
{
    // Earth feature. All attributes are relevant for this one planet.
    {
        geoneric::Feature earth;
        BOOST_CHECK_EQUAL(earth.nr_attributes(), 0u);

        earth.add_attribute("acceleration",
            std::make_shared<geoneric::ConstantAttribute<double>>(9.80665));
        BOOST_CHECK_EQUAL(earth.nr_attributes(), 1u);
    }

    // Planets feature. Attributes are stored per planet (a point in space).
    {
        geoneric::Feature planets;

        typedef int Value;
        typedef geoneric::Point<int, 3> Point;
        typedef geoneric::SpatialAttribute<geoneric::SpatialDomain<Point>,
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
