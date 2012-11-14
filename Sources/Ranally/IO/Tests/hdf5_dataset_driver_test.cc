#define BOOST_TEST_MODULE ranally io
#include <boost/test/included/unit_test.hpp>
#include "Ranally/Util/String.h"
#include "Ranally/IO/HDF5DatasetDriver.h"
#include "Ranally/IO/PointAttribute.h"
#include "Ranally/IO/PointDomain.h"
#include "Ranally/IO/PointFeature.h"


void removeTestFiles()
{
    std::vector<ranally::String> dataSetNames;
    dataSetNames.push_back("TestExists.h5");
    dataSetNames.push_back("TestCreate.h5");
    dataSetNames.push_back("TestRemove.h5");
    dataSetNames.push_back("TestOpen.h5");
    ranally::HDF5DatasetDriver driver;

    for(auto dataSetName: dataSetNames) {
        if(driver.exists(dataSetName)) {
            driver.remove(dataSetName);
        }
        assert(!driver.exists(dataSetName));
    }
}


BOOST_AUTO_TEST_SUITE(hdf5_dataset_driver)

BOOST_AUTO_TEST_CASE(exists)
{
    ranally::HDF5DatasetDriver driver;

    BOOST_REQUIRE(!driver.exists("TestExists.h5"));
    std::unique_ptr<ranally::HDF5Dataset>(driver.create("TestExists.h5"));
    BOOST_CHECK(driver.exists("TestExists.h5"));
}


BOOST_AUTO_TEST_CASE(create)
{
    // TODO Crashes.
    return;

    ranally::HDF5DatasetDriver driver;
    ranally::String dataSetName = "TestCreate.h5";
    BOOST_REQUIRE(!driver.exists(dataSetName));
    std::unique_ptr<ranally::HDF5Dataset> dataSet;

    // Create empty data set.
    {
        dataSet.reset();
        dataSet.reset(driver.create(dataSetName));
        BOOST_CHECK(driver.exists(dataSetName));

        dataSet.reset();
        dataSet.reset(driver.open(dataSetName));

        BOOST_CHECK(dataSet);
        BOOST_CHECK(dataSet->name() == dataSetName);
        BOOST_CHECK_EQUAL(dataSet->nrFeatures(), 0u);
    }

    // Create a data set with a feature without attributes.
    {
        ranally::PointsPtr points(new ranally::Points);
        points->push_back(ranally::Point(3.0, 4.0));
        ranally::PointDomainPtr domain(new ranally::PointDomain(points));
        ranally::PointFeature featureWritten("Stations", domain);

        dataSet.reset();
        dataSet.reset(driver.create(dataSetName));
        dataSet->addFeature(featureWritten);

        dataSet.reset();
        dataSet.reset(driver.open(dataSetName));
        BOOST_CHECK_EQUAL(dataSet->nrFeatures(), 1u);
        BOOST_CHECK(dataSet->exists("Stations"));

        ranally::PointFeaturePtr featureRead(
            dynamic_cast<ranally::PointFeature*>(dataSet->feature("Stations")));
        // TODO BOOST_CHECK(*featureRead == featureWritten);
        // BOOST_CHECK_EQUAL(featureRead->attributes().size(), 0u);
    }

    // Add a feature with an attribute.
    {
        ranally::PointsPtr points(new ranally::Points);
        points->push_back(ranally::Point(3.0, 4.0));
        ranally::PointDomainPtr domain(new ranally::PointDomain(points));;
        ranally::PointAttributePtr attribute(new ranally::PointAttribute(
          "Measuring", domain));
        ranally::PointFeature featureWritten("Stations", domain);
        featureWritten.add(attribute);

        dataSet.reset();
        dataSet.reset(driver.create(dataSetName));
        dataSet->addFeature(featureWritten);

        dataSet.reset();
        dataSet.reset(driver.open(dataSetName));
        BOOST_CHECK_EQUAL(dataSet->nrFeatures(), 1u);
        BOOST_CHECK(dataSet->exists("Stations"));

        ranally::PointFeaturePtr featureRead(
            dynamic_cast<ranally::PointFeature*>(dataSet->feature("Stations")));
        // TODO BOOST_CHECK(*featureRead == featureWritten);
        // BOOST_CHECK_EQUAL(featureRead->attributes().size(), 1u);
        // BOOST_CHECK(featureRead->exists("Measuring"));
    }


    // TODO Test creation of existing file.
    // TODO Test creation of unwritable file.
    // TODO Test creation of non existing path.
    // TODO Test creation of Unicode path.
}


BOOST_AUTO_TEST_CASE(remove)
{
    ranally::HDF5DatasetDriver driver;
    ranally::String dataSetName = "TestRemove.h5";
    BOOST_REQUIRE(!driver.exists(dataSetName));

    std::unique_ptr<ranally::HDF5Dataset>(driver.create(dataSetName));
    BOOST_CHECK(driver.exists(dataSetName));

    driver.remove(dataSetName);
    BOOST_CHECK(!driver.exists(dataSetName));

    // TODO Test remove of read-only file.
    // TODO Test remove of non-existing file.
}


BOOST_AUTO_TEST_CASE(open)
{
    ranally::HDF5DatasetDriver driver;
    ranally::String dataSetName = "TestOpen.h5";
    BOOST_REQUIRE(!driver.exists(dataSetName));

    std::unique_ptr<ranally::HDF5Dataset>(driver.create(dataSetName));
    BOOST_REQUIRE(driver.exists(dataSetName));

    std::unique_ptr<ranally::HDF5Dataset> dataSet(driver.open(
      dataSetName));
    BOOST_CHECK(dataSet);
    BOOST_CHECK(dataSet->name() == dataSetName);

    // TODO Test opening non-existing data set.
    // TODO Test opening read-only data set.
    // TODO Test opening write-only data set.
    // TODO Test opening executable-only data set.
}

BOOST_AUTO_TEST_SUITE_END()
