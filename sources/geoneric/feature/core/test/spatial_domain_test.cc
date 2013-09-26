#define BOOST_TEST_MODULE geoneric feature
#include <boost/test/unit_test.hpp>
#include "geoneric/feature/core/box.h"
#include "geoneric/feature/core/point.h"
#include "geoneric/feature/core/spatial_domain.h"


template<
    class CoordinateType,
    size_t nr_dimensions,
    class CoordinateSystem>
inline bool operator==(
    geoneric::Point<CoordinateType, nr_dimensions,
        CoordinateSystem> const& lhs,
    geoneric::Point<CoordinateType, nr_dimensions,
        CoordinateSystem> const& rhs)
{
    return
        geoneric::get<0>(lhs) == geoneric::get<0>(rhs) &&
        geoneric::get<1>(lhs) == geoneric::get<1>(rhs);
}


template<
    class Point>
inline bool operator==(
    geoneric::Box<Point> const& lhs,
    geoneric::Box<Point> const& rhs)
{
    return
        lhs.min_corner() == rhs.min_corner() &&
        lhs.max_corner() == rhs.max_corner();
}


BOOST_AUTO_TEST_SUITE(spatial_domain)

BOOST_AUTO_TEST_CASE(spatial_domain)
{
    typedef geoneric::Point<double, 2> Point;
    typedef geoneric::Box<Point> Box;
    typedef geoneric::SpatialDomain<Box> BoxDomain;

    BoxDomain spatial_domain;

    Point south_west;
    Point north_east;
    geoneric::set<0>(south_west, 1.1);
    geoneric::set<1>(south_west, 2.2);
    geoneric::set<0>(north_east, 3.3);
    geoneric::set<1>(north_east, 4.4);
    Box box1(south_west, north_east);
    Box box2(south_west, north_east);

    // Whatever the geometry, adding one to a domain should result in a
    // unique geometry id.
    BoxDomain::GID gid1 = spatial_domain.add(box1);
    BoxDomain::GID gid2 = spatial_domain.add(box2);
    BOOST_CHECK_NE(gid1, gid2);

    // Test geometries themselves.
    BOOST_CHECK(spatial_domain.geometry(gid1) == box1);
    BOOST_CHECK(spatial_domain.geometry(gid2) == box2);
}

BOOST_AUTO_TEST_SUITE_END()
