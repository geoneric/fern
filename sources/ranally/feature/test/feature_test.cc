#define BOOST_TEST_MODULE ranally feature
#include <boost/test/unit_test.hpp>
// #include <boost/geometry/algorithms/assign.hpp>
// #include <boost/geometry/geometry.hpp>
// #include <boost/geometry/geometries/adapted/c_array.hpp>
#include "ranally/core/string.h"
#include "ranally/feature/attribute.h"
#include "ranally/feature/domain_value.h"
#include "ranally/feature/feature.h"
#include "ranally/feature/feature_domain.h"
// #include "ranally/feature/point_domain.h"
// #include "ranally/feature/polygon_domain.h"
#include "ranally/feature/scalar_attribute.h"
#include "ranally/feature/scalar_domain.h"
#include "ranally/feature/scalar_value.h"

// BOOST_GEOMETRY_REGISTER_C_ARRAY_CS(cs::cartesian)


BOOST_AUTO_TEST_SUITE(feature)

BOOST_AUTO_TEST_CASE(scalar_feature_value_attributes)
{
    using namespace ranally;

    // Let's play with feature earth, being a scalar feature (whose location
    // in space and time is irrelevant).
    Feature earth(std::make_shared<ScalarDomain>());

    // Add some value attributes.

    // standard gravity, double.
    auto gravity_value(std::make_shared<ScalarValue<double>>(9.8));
    auto gravity_attribute(std::make_shared<ScalarAttribute<double>>("gravity",
        earth.domain<ScalarDomain>(), gravity_value));
    earth.add_attribute(gravity_attribute);

    // latin name, string.
    auto latin_name_value(std::make_shared<ScalarValue<String>>("terra"));
    auto latin_name_attribute(std::make_shared<ScalarAttribute<String>>(
        "latin_name", earth.domain<ScalarDomain>(), latin_name_value));
    earth.add_attribute(latin_name_attribute);

    // Test the earth feature's attributes.
    BOOST_CHECK_EQUAL(earth.nr_attributes(), 2u);

    // standard gravity.
    gravity_attribute = earth.attribute<ScalarAttribute<double>>("gravity");
    BOOST_REQUIRE(gravity_attribute);
    gravity_value = gravity_attribute->value();
    BOOST_CHECK_CLOSE((*gravity_value)(), 9.8, 0.001);

    // latin name
    latin_name_attribute = earth.attribute<ScalarAttribute<String>>(
        "latin_name");
    BOOST_REQUIRE(latin_name_attribute);
    latin_name_value = latin_name_attribute->value();
    BOOST_CHECK_EQUAL((*latin_name_value)(), "terra");
}


BOOST_AUTO_TEST_CASE(point_feature_value_attributes)
{
    using namespace ranally;

    // Let's play with the police car feature, being a point feature with some
    // value attributes.

    // We have four points representing four police cars.
    auto domain(std::make_shared<FeatureDomain<Point>>());
    domain->append(Point(1.0, 1.0));
    domain->append(Point(1.0, 2.0));
    domain->append(Point(2.0, 2.0));
    domain->append(Point(2.0, 1.0));

    // The feature contains the domain and no attributes.
    Feature police_car(domain);

    // Add some value attributes.

    // id, uint8
    auto id_value(std::make_shared<DomainValue<uint8_t>>(*domain));
    assert(id_value->size() == domain->geometry().size());
    auto& values((*id_value)());
    values[0] = 3u;
    values[1] = 5u;
    values[2] = 7u;
    values[3] = 9u;
    // auto id_attribute(std::make_shared<DomainAttribute<uint8_t>>("id", domain,
    //     id_value));
    // police_car.add_attribute(id_attribute);


    // Test the police_car's attributes.


}


// BOOST_AUTO_TEST_CASE(polygon_attributes)
// {
//     // namespace bg = boost::geometry;
// 
//     // // Let's play with feature earth.
//     // ranally::Feature earth;
// 
//     // // Add some polygon attributes.
//     // std::shared_ptr<ranally::Polygons> polygons(new ranally::Polygons());
//     // ranally::Polygon polygon;
// 
//     // {
//     //     polygon.clear();
//     //     auto& point_list(polygon.inners());
//     //     bg::append(point_list, bg::make<ranally::Point>(1.0, 1.0));
//     //     bg::append(point_list, bg::make<ranally::Point>(1.0, 2.0));
//     //     bg::append(point_list, bg::make<ranally::Point>(2.0, 2.0));
//     //     bg::append(point_list, bg::make<ranally::Point>(2.0, 1.0));
//     // }
//     // polygons->push_back(polygon);
// 
//     // {
//     //     polygon.clear();
//     //     auto& point_list(polygon.inners());
//     //     bg::append(point_list, bg::make<ranally::Point>(3.0, 1.0));
//     //     bg::append(point_list, bg::make<ranally::Point>(3.0, 2.0));
//     //     bg::append(point_list, bg::make<ranally::Point>(4.0, 2.0));
//     //     bg::append(point_list, bg::make<ranally::Point>(4.0, 1.0));
//     // }
//     // polygons->push_back(polygon);
// 
//     // std::shared_ptr<ranally::PolygonDomain> polygon_domain(
//     //     new ranally::PolygonDomain(polygons));
// 
//     // // TODO Besided the coordinates, each geometry must have an id.
// 
//     // // name of each continent, string
//     // // TODO ...
// 
//     // // Population of each continent, unsigned integral.
//     // // TODO ...
// 
// 
// }

BOOST_AUTO_TEST_SUITE_END()
