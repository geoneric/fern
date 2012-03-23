#include "Ranally/IO/HDF5DataSetDriverTest.h"
#include <boost/foreach.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>
#include "Ranally/Util/String.h"
#include "Ranally/IO/HDF5DataSetDriver.h"
#include "Ranally/IO/PointAttribute.h"
#include "Ranally/IO/PointDomain.h"
#include "Ranally/IO/PointFeature.h"



boost::unit_test::test_suite* HDF5DataSetDriverTest::suite()
{
  boost::unit_test::test_suite* suite = BOOST_TEST_SUITE(__FILE__);
  boost::shared_ptr<HDF5DataSetDriverTest> instance(
    new HDF5DataSetDriverTest());
  suite->add(BOOST_CLASS_TEST_CASE(
    &HDF5DataSetDriverTest::testExists, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
    &HDF5DataSetDriverTest::testCreate, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
    &HDF5DataSetDriverTest::testRemove, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
    &HDF5DataSetDriverTest::testOpen, instance));

  return suite;
}



HDF5DataSetDriverTest::HDF5DataSetDriverTest()
{
  removeTestFiles();
}



HDF5DataSetDriverTest::~HDF5DataSetDriverTest()
{
  // removeTestFiles();
}



void HDF5DataSetDriverTest::removeTestFiles()
{
  std::vector<UnicodeString> dataSetNames;
  dataSetNames.push_back("TestExists.h5");
  dataSetNames.push_back("TestCreate.h5");
  dataSetNames.push_back("TestRemove.h5");
  dataSetNames.push_back("TestOpen.h5");
  ranally::io::HDF5DataSetDriver driver;

  BOOST_FOREACH(UnicodeString const& dataSetName, dataSetNames) {
    if(driver.exists(dataSetName)) {
      driver.remove(dataSetName);
    }
    assert(!driver.exists(dataSetName));
  }
}



void HDF5DataSetDriverTest::testExists()
{
  ranally::io::HDF5DataSetDriver driver;

  BOOST_REQUIRE(!driver.exists("TestExists.h5"));
  boost::scoped_ptr<ranally::io::HDF5DataSet>(driver.create("TestExists.h5"));
  BOOST_CHECK(driver.exists("TestExists.h5"));
}



void HDF5DataSetDriverTest::testCreate()
{
  ranally::io::HDF5DataSetDriver driver;
  UnicodeString dataSetName = "TestCreate.h5";
  BOOST_REQUIRE(!driver.exists(dataSetName));
  boost::scoped_ptr<ranally::io::HDF5DataSet> dataSet;

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

    // ranally::PointFeaturePtr featureRead(dynamic_cast<ranally::PointFeature*>(
    //   dataSet->feature("Stations")));
    // // TODO BOOST_CHECK(*featureRead == featureWritten);
    // // BOOST_CHECK_EQUAL(featureRead->attributes().size(), 0u);
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
    // dataSet->addFeature(featureWritten);

    dataSet.reset();
    // dataSet.reset(driver.open(dataSetName));
    // BOOST_CHECK_EQUAL(dataSet->nrFeatures(), 1u);
    // BOOST_CHECK(dataSet->exists("Stations"));

    // // ranally::PointFeaturePtr featureRead(dynamic_cast<ranally::PointFeature*>(
    // //   dataSet->feature("Stations")));
    // // TODO BOOST_CHECK(*featureRead == featureWritten);
    // // BOOST_CHECK_EQUAL(featureRead->attributes().size(), 1u);
    // // BOOST_CHECK(featureRead->exists("Measuring"));
  }


  // TODO Test creation of existing file.
  // TODO Test creation of unwritable file.
  // TODO Test creation of non existing path.
  // TODO Test creation of Unicode path.
}



void HDF5DataSetDriverTest::testRemove()
{
  ranally::io::HDF5DataSetDriver driver;
  UnicodeString dataSetName = "TestRemove.h5";
  BOOST_REQUIRE(!driver.exists(dataSetName));

  boost::scoped_ptr<ranally::io::HDF5DataSet>(driver.create(dataSetName));
  BOOST_CHECK(driver.exists(dataSetName));

  driver.remove(dataSetName);
  BOOST_CHECK(!driver.exists(dataSetName));

  // TODO Test remove of read-only file.
  // TODO Test remove of non-existing file.
}



void HDF5DataSetDriverTest::testOpen()
{
  ranally::io::HDF5DataSetDriver driver;
  UnicodeString dataSetName = "TestOpen.h5";
  BOOST_REQUIRE(!driver.exists(dataSetName));

  boost::scoped_ptr<ranally::io::HDF5DataSet>(driver.create(dataSetName));
  BOOST_REQUIRE(driver.exists(dataSetName));

  boost::scoped_ptr<ranally::io::HDF5DataSet> dataSet(driver.open(
    dataSetName));
  BOOST_CHECK(dataSet);
  BOOST_CHECK(dataSet->name() == dataSetName);

  // TODO Test opening non-existing data set.
  // TODO Test opening read-only data set.
  // TODO Test opening write-only data set.
  // TODO Test opening executable-only data set.
}

