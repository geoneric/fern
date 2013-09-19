#define BOOST_TEST_MODULE geoneric io
#include <boost/test/unit_test.hpp>
#include "geoneric/io/gdal/constant_attribute.h"
#include "geoneric/io/gdal/feature.h"
#include "geoneric/io/gdal/point.h"
#include "geoneric/io/gdal/spatial_attribute.h"
#include "geoneric/io/gdal/spatial_domain.h"


BOOST_AUTO_TEST_SUITE(feature)

BOOST_AUTO_TEST_CASE(feature)
{
    // Earth feature. All attributes are relevant for this one planet.
    {
        geoneric::Feature earth;
        BOOST_CHECK(earth.empty());

        earth["acceleration"] =
            std::make_shared<geoneric::ConstantAttribute<double>>(9.80665);
        BOOST_CHECK(!earth.empty());
        BOOST_CHECK_EQUAL(earth.size(), 1);
    }

    // Planets feature. Attributes are stored per planet (a point in space).
    {
        geoneric::Feature planets;

        typedef geoneric::Point<int, 3> Point;
        typedef geoneric::SpatialDomain<Point> PointDomain;
        typedef int Value;
        typedef geoneric::SpatialAttribute<PointDomain, Value> PointsAttribute;
        typedef std::shared_ptr<PointsAttribute> PointsAttributePtr;

        Point center_of_planet1 = { 1, 1, 1 };
        PointsAttributePtr acceleration(new PointsAttribute());
        acceleration->add(center_of_planet1, 1.1);
        planets["acceleration"] = acceleration;
        BOOST_CHECK(!planets.empty());
        BOOST_CHECK_EQUAL(planets.size(), 1);
    }
}

BOOST_AUTO_TEST_SUITE_END()
