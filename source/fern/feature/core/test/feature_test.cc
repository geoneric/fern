// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
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

        using Value = int;
        using Point = fern::Point<int, 3>;
        using PointsAttribute = fern::SpatialAttribute<fern::SpatialDomain<
            Point>, Value>;
        using PointsAttributePtr = std::shared_ptr<PointsAttribute>;

        Point center_of_planet1 = { 1, 1, 1 };
        PointsAttributePtr acceleration(std::make_shared<PointsAttribute>());
        acceleration->add(center_of_planet1, 5);
        planets.add_attribute("acceleration", acceleration);
        BOOST_CHECK_EQUAL(planets.nr_attributes(), 1);
    }
}

BOOST_AUTO_TEST_SUITE_END()
