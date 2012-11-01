#include "OGRDataSetDriverTest.h"
#include <memory>
#include <boost/shared_ptr.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>
#include "Ranally/Util/String.h"
#include "Ranally/IO/OGRDataSetDriver.h"
#include "Ranally/IO/PointDomain.h"
#include "Ranally/IO/PointFeature.h"


boost::unit_test::test_suite* OGRDataSetDriverTest::suite()
{
    boost::unit_test::test_suite* suite = BOOST_TEST_SUITE(__FILE__);
    boost::shared_ptr<OGRDataSetDriverTest> instance(
        new OGRDataSetDriverTest());
    suite->add(BOOST_CLASS_TEST_CASE(
        &OGRDataSetDriverTest::testExists, instance));
    suite->add(BOOST_CLASS_TEST_CASE(
        &OGRDataSetDriverTest::testCreate, instance));
    suite->add(BOOST_CLASS_TEST_CASE(
        &OGRDataSetDriverTest::testRemove, instance));
    suite->add(BOOST_CLASS_TEST_CASE(
        &OGRDataSetDriverTest::testOpen, instance));

    return suite;
}


OGRDataSetDriverTest::OGRDataSetDriverTest()
{
    removeTestFiles();
}


OGRDataSetDriverTest::~OGRDataSetDriverTest()
{
    // OGRClient is destructed by the time this runs.
    // removeTestFiles();
}


void OGRDataSetDriverTest::removeTestFiles()
{
    std::vector<ranally::String> dataSetNames;
    dataSetNames.push_back("TestCreate.shp");
    dataSetNames.push_back("TestRemove.shp");
    ranally::OGRDataSetDriver driver("ESRI Shapefile");

    for(auto dataSetName: dataSetNames) {
        if(driver.exists(dataSetName)) {
            driver.remove(dataSetName);
        }
        assert(!driver.exists(dataSetName));
    }
}


void OGRDataSetDriverTest::testExists()
{
    ranally::OGRDataSetDriver driver("GeoJSON");

    BOOST_REQUIRE(!driver.exists("DoesNotExist"));
    BOOST_REQUIRE( driver.exists("Point.json"));
    BOOST_REQUIRE( driver.exists("ReadOnlyPoint.json"));
    BOOST_REQUIRE(!driver.exists("WriteOnlyPoint.json"));
}


void OGRDataSetDriverTest::testCreate()
{
    ranally::OGRDataSetDriver driver("ESRI Shapefile");
    std::unique_ptr<ranally::OGRDataSet> dataSet;

    // Test creation of new data set. ------------------------------------------
    ranally::String dataSetName = "TestCreate.shp";
    BOOST_REQUIRE(!driver.exists(dataSetName));

    dataSet.reset(driver.create(dataSetName));
    // Since there are no layers in the data set, OGR hasn't written the data set
    // to file yet.
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


void OGRDataSetDriverTest::testRemove()
{
    ranally::OGRDataSetDriver driver("ESRI Shapefile");
    std::unique_ptr<ranally::OGRDataSet> dataSet;
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
        ranally::OGRDataSetDriver driver("GeoJSON");
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


void OGRDataSetDriverTest::testOpen()
{
    ranally::OGRDataSetDriver driver("GeoJSON");
    std::unique_ptr<ranally::OGRDataSet> dataSet;
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
