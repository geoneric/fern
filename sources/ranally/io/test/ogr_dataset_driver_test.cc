#define BOOST_TEST_MODULE ranally io
#include <boost/test/included/unit_test.hpp>
#include "ranally/core/string.h"
#include "ranally/io/ogr_client.h"
#include "ranally/io/ogr_dataset_driver.h"
#include "ranally/io/point_domain.h"
#include "ranally/io/point_feature.h"


void remove_test_files()
{
    std::vector<ranally::String> dataset_names;
    dataset_names.push_back("TestCreate.shp");
    dataset_names.push_back("TestRemove.shp");
    ranally::OGRDatasetDriver driver("ESRI Shapefile");

    for(auto dataset_name: dataset_names) {
        if(driver.exists(dataset_name)) {
            driver.remove(dataset_name);
        }
        assert(!driver.exists(dataset_name));
    }
}


class Support:
    public ranally::OGRClient
{

public:

    Support()
        : ranally::OGRClient()
    {
        remove_test_files();
    }

};


BOOST_FIXTURE_TEST_SUITE(ogr_dataset_driver, Support)

BOOST_AUTO_TEST_CASE(exists)
{
    ranally::OGRDatasetDriver driver("GeoJSON");

    BOOST_REQUIRE(!driver.exists("DoesNotExist"));
    BOOST_REQUIRE( driver.exists("Point.json"));
    BOOST_REQUIRE( driver.exists("ReadOnlyPoint.json"));
    BOOST_REQUIRE(!driver.exists("WriteOnlyPoint.json"));
}


BOOST_AUTO_TEST_CASE(create)
{
    ranally::OGRDatasetDriver driver("ESRI Shapefile");
    std::unique_ptr<ranally::OGRDataset> dataSet;

    // Test creation of new data set. ------------------------------------------
    ranally::String dataset_name = "TestCreate.shp";
    BOOST_REQUIRE(!driver.exists(dataset_name));

    dataSet.reset(driver.create(dataset_name));
    // Since there are no layers in the data set, OGR hasn't written the data
    // set to file yet.
    BOOST_CHECK(!driver.exists(dataset_name));

    // Add a feature.
    ranally::PointsPtr points(new ranally::Points);
    points->push_back(ranally::Point(3.0, 4.0));
    ranally::PointDomainPtr domain(new ranally::PointDomain(points));;
    ranally::PointFeature feature("Stations", domain);
    dataSet->add_feature(feature);

    // Now it should work.
    BOOST_CHECK(driver.exists(dataset_name));
    BOOST_CHECK(dataSet);
    BOOST_CHECK(dataSet->name() == dataset_name);

    // Test creation of existing file. -----------------------------------------
    BOOST_CHECK(driver.exists(dataset_name));

    dataSet.reset(driver.create(dataset_name));
    // Since there are no layers in the data set, OGR hasn't written the data
    // set to file yet. The original data set should have been deleted though.
    BOOST_CHECK(!driver.exists(dataset_name));

    // The rest of the test would equal the stuff above...

    // Test creation of unwritable file. ---------------------------------------
    dataset_name = "ReadOnlyDir/TestCreate.shp";
    BOOST_CHECK(!driver.exists(dataset_name));

    // This succeeds since no feature layer have been created yet.
    dataSet.reset(driver.create(dataset_name));

    // TODO Exception.
    BOOST_CHECK_THROW(dataSet->add_feature(feature), std::string);

    // Test creation of non existing path. -------------------------------------
    dataset_name = "DoesNotExist/TestCreate.shp";
    BOOST_CHECK(!driver.exists(dataset_name));

    // This succeeds since no feature layer have been created yet.
    dataSet.reset(driver.create(dataset_name));

    // TODO Exception.
    BOOST_CHECK_THROW(dataSet->add_feature(feature), std::string);

    // TODO Test creation of Unicode path.
}


BOOST_AUTO_TEST_CASE(remove)
{
    ranally::OGRDatasetDriver driver("ESRI Shapefile");
    std::unique_ptr<ranally::OGRDataset> dataSet;
    ranally::String dataset_name;

    dataset_name = "TestRemove.shp";
    BOOST_REQUIRE(!driver.exists(dataset_name));

    dataSet.reset(driver.create(dataset_name));
    BOOST_CHECK(!driver.exists(dataset_name));
    ranally::PointsPtr points(new ranally::Points);
    points->push_back(ranally::Point(3.0, 4.0));
    ranally::PointDomainPtr domain(new ranally::PointDomain(points));;
    ranally::PointFeature feature("Stations", domain);
    dataSet->add_feature(feature);
    BOOST_CHECK(driver.exists(dataset_name));

    driver.remove(dataset_name);
    BOOST_CHECK(!driver.exists(dataset_name));

    {
        // Test remove of read-only file. --------------------------------------
        ranally::OGRDatasetDriver driver("GeoJSON");
        dataset_name = "ReadOnlyPoint.json";
        BOOST_CHECK(driver.exists(dataset_name));
        // TODO Read-only file is removed... Report to gdal list?!
        BOOST_WARN_THROW(driver.remove(dataset_name), std::string);

        // Test remove of non-existing file. -----------------------------------
        dataset_name = "DoesNotExist.json";
        BOOST_CHECK(!driver.exists(dataset_name));
        BOOST_CHECK_THROW(driver.remove(dataset_name), std::string);
    }
}


BOOST_AUTO_TEST_CASE(open)
{
    ranally::OGRDatasetDriver driver("GeoJSON");
    std::unique_ptr<ranally::OGRDataset> dataSet;
    ranally::String dataset_name;

    dataset_name = "Point.json";
    BOOST_REQUIRE(driver.exists(dataset_name));
    dataSet.reset(driver.open(dataset_name));
    BOOST_CHECK(dataSet);
    BOOST_CHECK(dataSet->name() == dataset_name);

    // Test opening non-existing data set. -------------------------------------
    BOOST_CHECK_THROW(driver.open("DoesNotExist.json"), std::string);

    // Test opening read-only data set. ----------------------------------------
    BOOST_CHECK_THROW(driver.open("ReadOnlyPoint.json"), std::string);
    BOOST_REQUIRE(driver.exists(dataset_name));
    dataSet.reset(driver.open(dataset_name));
    BOOST_CHECK(dataSet);
    BOOST_CHECK(dataSet->name() == dataset_name);

    // Test opening write-only data set. ---------------------------------------
    BOOST_CHECK_THROW(driver.open("WriteOnlyPoint.json"), std::string);

    // Test opening executable-only data set. ----------------------------------
    BOOST_CHECK_THROW(driver.open("ExecutableOnlyPoint.json"), std::string);
}

BOOST_AUTO_TEST_SUITE_END()
