#include "Ranally/IO/HDF5DataSetDriverTest.h"
#include <boost/foreach.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>
#include "Ranally/Util/String.h"
#include "Ranally/IO/HDF5DataSetDriver.h"



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
  removeTestFiles();
}



void HDF5DataSetDriverTest::removeTestFiles()
{
  std::vector<UnicodeString> dataSetNames;
  dataSetNames.push_back("TestExists.hdf5");
  dataSetNames.push_back("TestCreate.hdf5");
  dataSetNames.push_back("TestRemove.hdf5");
  dataSetNames.push_back("TestOpen.hdf5");
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
  UnicodeString dataSetName = "TestExists.hdf5";
  BOOST_REQUIRE(!driver.exists(dataSetName));

  (void)driver.create(dataSetName);
  BOOST_CHECK(driver.exists(dataSetName));
}



void HDF5DataSetDriverTest::testCreate()
{
  ranally::io::HDF5DataSetDriver driver;
  UnicodeString dataSetName = "TestCreate.hdf5";
  BOOST_REQUIRE(!driver.exists(dataSetName));

  boost::scoped_ptr<ranally::io::HDF5DataSet> dataSet(driver.create(
    dataSetName));
  BOOST_CHECK(driver.exists(dataSetName));
  BOOST_CHECK(dataSet);
  BOOST_CHECK(dataSet->name() == dataSetName);

  // TODO Test creation of existing file.
  // TODO Test creation of unwritable file.
  // TODO Test creation of non existing path.
  // TODO Test creation of Unicode path.
}



void HDF5DataSetDriverTest::testRemove()
{
  ranally::io::HDF5DataSetDriver driver;
  UnicodeString dataSetName = "TestRemove.hdf5";
  BOOST_REQUIRE(!driver.exists(dataSetName));

  (void)driver.create(dataSetName);
  BOOST_CHECK(driver.exists(dataSetName));

  driver.remove(dataSetName);
  BOOST_CHECK(!driver.exists(dataSetName));

  // TODO Test remove of read-only file.
  // TODO Test remove of non-existing file.
}



void HDF5DataSetDriverTest::testOpen()
{
  ranally::io::HDF5DataSetDriver driver;
  UnicodeString dataSetName = "TestOpen.hdf5";
  BOOST_REQUIRE(!driver.exists(dataSetName));

  (void)driver.create(dataSetName);
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

