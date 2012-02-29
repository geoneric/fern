#include "OGRDataSetDriverTest.h"
#include <boost/foreach.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>
#include "Ranally/Util/String.h"
#include "Ranally/IO/OGRDataSetDriver.h"



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
  removeTestFiles();
}



void OGRDataSetDriverTest::removeTestFiles()
{
  std::vector<UnicodeString> dataSetNames;
  dataSetNames.push_back("TestCreate.shp");
  dataSetNames.push_back("TestRemove.shp");
  ranally::io::OGRDataSetDriver driver;

  BOOST_FOREACH(UnicodeString const& dataSetName, dataSetNames) {
    if(driver.exists(dataSetName)) {
      driver.remove(dataSetName);
    }
    assert(!driver.exists(dataSetName));
  }
}



void OGRDataSetDriverTest::testExists()
{
  ranally::io::OGRDataSetDriver driver;
  UnicodeString dataSetName;

  {
    dataSetName = "DoesNotExist";
    BOOST_REQUIRE(!driver.exists(dataSetName));
  }

  {
    dataSetName = "Point.json";
    BOOST_REQUIRE(driver.exists(dataSetName));
  }
}



void OGRDataSetDriverTest::testCreate()
{
  ranally::io::OGRDataSetDriver driver;
  UnicodeString dataSetName = "TestCreate.shp";
  BOOST_REQUIRE(!driver.exists(dataSetName));

  boost::scoped_ptr<ranally::io::OGRDataSet> dataSet(driver.create(
    dataSetName));
  // Since there are no layers in the data set, OGR hasn't written the data set
  // to file yet.
  // TODO Add some features.
  // BOOST_CHECK(driver.exists(dataSetName));
  BOOST_CHECK(dataSet);
  BOOST_CHECK(dataSet->name() == dataSetName);

  // // TODO Test creation of existing file.
  // // TODO Test creation of unwritable file.
  // // TODO Test creation of non existing path.
  // // TODO Test creation of Unicode path.
}



void OGRDataSetDriverTest::testRemove()
{
  // TODO
  // ranally::io::OGRDataSetDriver driver;
  // UnicodeString dataSetName = "TestRemove.hdf5";
  // BOOST_REQUIRE(!driver.exists(dataSetName));

  // (void)driver.create(dataSetName);
  // BOOST_CHECK(driver.exists(dataSetName));

  // driver.remove(dataSetName);
  // BOOST_CHECK(!driver.exists(dataSetName));

  // TODO Test remove of read-only file.
  // TODO Test remove of non-existing file.
}



void OGRDataSetDriverTest::testOpen()
{
  ranally::io::OGRDataSetDriver driver;
  UnicodeString dataSetName;

  {
    dataSetName = "Point.json";
    BOOST_REQUIRE(driver.exists(dataSetName));

    boost::scoped_ptr<ranally::io::OGRDataSet> dataSet(driver.open(
      dataSetName));
    BOOST_CHECK(dataSet);
    BOOST_CHECK(dataSet->name() == dataSetName);
  }

  // TODO Test opening non-existing data set.
  // TODO Test opening read-only data set.
  // TODO Test opening write-only data set.
  // TODO Test opening executable-only data set.
}

