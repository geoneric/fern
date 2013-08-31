#define BOOST_TEST_MODULE geoneric feature
#include <boost/test/unit_test.hpp>
// #include <boost/geometry/algorithms/assign.hpp>
// #include <boost/geometry/geometry.hpp>
// #include <boost/geometry/geometries/adapted/c_array.hpp>
#include "geoneric/core/string.h"
#include "geoneric/feature/attribute.h"
#include "geoneric/feature/domain_attribute.h"
#include "geoneric/feature/domain_value.h"
#include "geoneric/feature/feature.h"
#include "geoneric/feature/feature_domain.h"
// #include "geoneric/feature/feature_domain_value.h"
// #include "geoneric/feature/point_domain.h"
// #include "geoneric/feature/polygon_domain.h"
#include "geoneric/feature/scalar_attribute.h"
#include "geoneric/feature/scalar_domain.h"
#include "geoneric/feature/scalar_value.h"

// BOOST_GEOMETRY_REGISTER_C_ARRAY_CS(cs::cartesian)


BOOST_AUTO_TEST_SUITE(feature)

BOOST_AUTO_TEST_CASE(scalar_feature_value_attributes)
{
    using namespace geoneric;

    // Let's play with feature earth, being a scalar feature (whose location
    // in space and time is irrelevant).
    Feature earth(std::make_shared<ScalarDomain>());

    // Add some value attributes.

    // standard gravity, double.
    auto gravity_value(std::make_shared<ScalarValue<double>>(9.8));
    auto gravity_attribute(std::make_shared<ScalarAttribute<double>>(
        earth.domain<ScalarDomain>(), gravity_value));
    earth.add_attribute("gravity", gravity_attribute);

    // latin name, string.
    auto latin_name_value(std::make_shared<ScalarValue<String>>("terra"));
    auto latin_name_attribute(std::make_shared<ScalarAttribute<String>>(
        earth.domain<ScalarDomain>(), latin_name_value));
    earth.add_attribute("latin_name", latin_name_attribute);

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
    using namespace geoneric;

    // Let's play with the police car feature, being a point feature with some
    // value attributes.

    // We have four points representing four police cars.
    auto domain(std::make_shared<FeatureDomain<Point>>());
    domain->insert(1, Point(1.0, 1.0));
    domain->insert(2, Point(1.0, 2.0));
    domain->insert(3, Point(2.0, 2.0));
    domain->insert(4, Point(2.0, 1.0));

    // The feature contains the domain and no attributes.
    Feature police_car(domain);

    // Add some value attributes.

    // id, uint8
    auto id_value(std::make_shared<DomainValue<uint8_t>>(*domain));
    BOOST_REQUIRE_EQUAL(id_value->size(), domain->size());
    id_value->insert(1, 3u);
    id_value->insert(2, 5u);
    id_value->insert(3, 7u);
    id_value->insert(4, 9u);
    auto id_attribute(std::make_shared<DomainAttribute<Point, uint8_t>>(
        domain, id_value));
    police_car.add_attribute("id", id_attribute);

    // TODO More attributes...

    // Test the police_car's domain.
    domain = police_car.domain<FeatureDomain<Point>>();
    BOOST_CHECK_EQUAL(domain->size(), 4u);
    BOOST_CHECK_EQUAL(domain->at(1), Point(1.0, 1.0));
    BOOST_CHECK_EQUAL(domain->at(2), Point(1.0, 2.0));
    BOOST_CHECK_EQUAL(domain->at(3), Point(2.0, 2.0));
    BOOST_CHECK_EQUAL(domain->at(4), Point(2.0, 1.0));

    // Test the police_car's attributes.
    id_attribute = police_car.attribute<DomainAttribute<Point, uint8_t>>("id");
    BOOST_REQUIRE(id_attribute);
    id_value = id_attribute->value();
    BOOST_CHECK_EQUAL(id_value->size(), 4u);
    BOOST_CHECK_EQUAL(id_value->at(1), 3u);
    BOOST_CHECK_EQUAL(id_value->at(2), 5u);
    BOOST_CHECK_EQUAL(id_value->at(3), 7u);
    BOOST_CHECK_EQUAL(id_value->at(4), 9u);
}


BOOST_AUTO_TEST_CASE(point_feature_feature_attributes)
{
    using namespace geoneric;

    // Let's play with the deer car feature, being a point feature with some
    // feature attributes.

    // We have four points representing four deer cars.
    auto domain(std::make_shared<FeatureDomain<Point>>());
    domain->insert(1, Point(1.0, 1.0));
    domain->insert(2, Point(1.0, 2.0));
    domain->insert(3, Point(2.0, 2.0));
    domain->insert(4, Point(2.0, 1.0));

    // The feature contains the domain and no attributes.
    Feature deer(domain);

    // Add some feature attributes.

    // place_of_birth, point feature.
    // First create the place_of_birth feature, then add it as a value to the
    // deer feature.

    // auto place_of_birth(std::make_shared<FeatureDomainValue>(*domain));
    // BOOST_REQUIRE_EQUAL(place_of_birth->size(), domain->size());
    // place_of_birth->insert(1, 3u);
    // place_of_birth->insert(2, 5u);
    // place_of_birth->insert(3, 7u);
    // place_of_birth->insert(4, 9u);
    // auto id_attribute(std::make_shared<DomainAttribute<Point, uint8_t>>("id",
    //     domain, place_of_birth));
    // police_car.add_attribute(id_attribute);




}


// BOOST_AUTO_TEST_CASE(polygon_attributes)
// {
//     // namespace bg = boost::geometry;
// 
//     // // Let's play with feature earth.
//     // geoneric::Feature earth;
// 
//     // // Add some polygon attributes.
//     // std::shared_ptr<geoneric::Polygons> polygons(new geoneric::Polygons());
//     // geoneric::Polygon polygon;
// 
//     // {
//     //     polygon.clear();
//     //     auto& point_list(polygon.inners());
//     //     bg::append(point_list, bg::make<geoneric::Point>(1.0, 1.0));
//     //     bg::append(point_list, bg::make<geoneric::Point>(1.0, 2.0));
//     //     bg::append(point_list, bg::make<geoneric::Point>(2.0, 2.0));
//     //     bg::append(point_list, bg::make<geoneric::Point>(2.0, 1.0));
//     // }
//     // polygons->push_back(polygon);
// 
//     // {
//     //     polygon.clear();
//     //     auto& point_list(polygon.inners());
//     //     bg::append(point_list, bg::make<geoneric::Point>(3.0, 1.0));
//     //     bg::append(point_list, bg::make<geoneric::Point>(3.0, 2.0));
//     //     bg::append(point_list, bg::make<geoneric::Point>(4.0, 2.0));
//     //     bg::append(point_list, bg::make<geoneric::Point>(4.0, 1.0));
//     // }
//     // polygons->push_back(polygon);
// 
//     // std::shared_ptr<geoneric::PolygonDomain> polygon_domain(
//     //     new geoneric::PolygonDomain(polygons));
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
