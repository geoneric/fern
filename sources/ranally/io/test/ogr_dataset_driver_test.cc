#define BOOST_TEST_MODULE ranally io
#include <boost/test/included/unit_test.hpp>
#include "ranally/core/string.h"
#include "ranally/io/ogr_client.h"
#include "ranally/io/ogr_dataset_driver.h"
#include "ranally/io/point_domain.h"
#include "ranally/io/point_feature.h"


void removeTestFiles()
{
    std::vector<ranally::String> dataSetNames;
    dataSetNames.push_back("TestCreate.shp");
    dataSetNames.push_back("TestRemove.shp");
    ranally::OGRDatasetDriver driver("ESRI Shapefile");

    for(auto dataSetName: dataSetNames) {
        if(driver.exists(dataSetName)) {
            driver.remove(dataSetName);
        }
        assert(!driver.exists(dataSetName));
    }
}


class Support:
    public ranally::OGRClient
{

public:

    Support()
        : ranally::OGRClient()
    {
        removeTestFiles();
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
    ranally::String dataSetName = "TestCreate.shp";
    BOOST_REQUIRE(!driver.exists(dataSetName));

    dataSet.reset(driver.create(dataSetName));
    // Since there are no layers in the data set, OGR hasn't written the data
    // set to file yet.
    BOOST_CHECK(!driver.exists(dataSetName));

    // Add a feature.
    ranally::PointsPtr points(new ranally::Points);
    points->push_back(ranally::Point(3.0, 4.0));
    ranally::PointDomainPtr domain(new ranally::PointDomain(points));;
    ranally::PointFeature feature("Stations", domain);
    dataSet->addFeature(feature);

    // Now it should work.
    BOOST_CHECK(driver.exists(dataSetName));
    BOOST_CHECK(dataSet);
    BOOST_CHECK(dataSet->name() == dataSetName);

    // Test creation of existing file. -----------------------------------------
    BOOST_CHECK(driver.exists(dataSetName));

    dataSet.reset(driver.create(dataSetName));
    // Since there are no layers in the data set, OGR hasn't written the data
    // set to file yet. The original data set should have been deleted though.
    BOOST_CHECK(!driver.exists(dataSetName));

    // The rest of the test would equal the stuff above...

    // Test creation of unwritable file. ---------------------------------------
    dataSetName = "ReadOnlyDir/TestCreate.shp";
    BOOST_CHECK(!driver.exists(dataSetName));

    // This succeeds since no feature layer have been created yet.
    dataSet.reset(driver.create(dataSetName));

    // TODO Exception.
    BOOST_CHECK_THROW(dataSet->addFeature(feature), std::string);

    // Test creation of non existing path. -------------------------------------
    dataSetName = "DoesNotExist/TestCreate.shp";
    BOOST_CHECK(!driver.exists(dataSetName));

    // This succeeds since no feature layer have been created yet.
    dataSet.reset(driver.create(dataSetName));

    // TODO Exception.
    BOOST_CHECK_THROW(dataSet->addFeature(feature), std::string);

    // TODO Test creation of Unicode path.
}


BOOST_AUTO_TEST_CASE(remove)
{
    ranally::OGRDatasetDriver driver("ESRI Shapefile");
    std::unique_ptr<ranally::OGRDataset> dataSet;
    ranally::String dataSetName;

    dataSetName = "TestRemove.shp";
    BOOST_REQUIRE(!driver.exists(dataSetName));

    dataSet.reset(driver.create(dataSetName));
    BOOST_CHECK(!driver.exists(dataSetName));
    ranally::PointsPtr points(new ranally::Points);
    points->push_back(ranally::Point(3.0, 4.0));
    ranally::PointDomainPtr domain(new ranally::PointDomain(points));;
    ranally::PointFeature feature("Stations", domain);
    dataSet->addFeature(feature);
    BOOST_CHECK(driver.exists(dataSetName));

    driver.remove(dataSetName);
    BOOST_CHECK(!driver.exists(dataSetName));

    {
        // Test remove of read-only file. --------------------------------------
        ranally::OGRDatasetDriver driver("GeoJSON");
        dataSetName = "ReadOnlyPoint.json";
        BOOST_CHECK(driver.exists(dataSetName));
        // TODO Read-only file is removed... Report to gdal list?!
        BOOST_WARN_THROW(driver.remove(dataSetName), std::string);

        // Test remove of non-existing file. -----------------------------------
        dataSetName = "DoesNotExist.json";
        BOOST_CHECK(!driver.exists(dataSetName));
        BOOST_CHECK_THROW(driver.remove(dataSetName), std::string);
    }
}


BOOST_AUTO_TEST_CASE(open)
{
    ranally::OGRDatasetDriver driver("GeoJSON");
    std::unique_ptr<ranally::OGRDataset> dataSet;
    ranally::String dataSetName;

    dataSetName = "Point.json";
    BOOST_REQUIRE(driver.exists(dataSetName));
    dataSet.reset(driver.open(dataSetName));
    BOOST_CHECK(dataSet);
    BOOST_CHECK(dataSet->name() == dataSetName);

    // Test opening non-existing data set. -------------------------------------
    BOOST_CHECK_THROW(driver.open("DoesNotExist.json"), std::string);

    // Test opening read-only data set. ----------------------------------------
    BOOST_CHECK_THROW(driver.open("ReadOnlyPoint.json"), std::string);
    BOOST_REQUIRE(driver.exists(dataSetName));
    dataSet.reset(driver.open(dataSetName));
    BOOST_CHECK(dataSet);
    BOOST_CHECK(dataSet->name() == dataSetName);

    // Test opening write-only data set. ---------------------------------------
    BOOST_CHECK_THROW(driver.open("WriteOnlyPoint.json"), std::string);

    // Test opening executable-only data set. ----------------------------------
    BOOST_CHECK_THROW(driver.open("ExecutableOnlyPoint.json"), std::string);
}

BOOST_AUTO_TEST_SUITE_END()
