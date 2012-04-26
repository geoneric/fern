#include <boost/test/included/unit_test.hpp>
#include "Ranally/IO/HDF5Client.h"
#include "Ranally/IO/HDF5DataSetDriverTest.h"
#include "Ranally/IO/HDF5DataSetTest.h"
#include "Ranally/IO/OGRClient.h"
#include "Ranally/IO/OGRDataSetDriverTest.h"



boost::unit_test::test_suite* init_unit_test_suite(
         int argc,
         char** const argv) {

  struct TestSuite:
    public boost::unit_test::test_suite,
    ranally::io::OGRClient,
    ranally::io::HDF5Client
  {
    TestSuite(
         int& /* argc */,
         char** /* argv */)
      : boost::unit_test::test_suite("Master test suite"),
        OGRClient(),
        HDF5Client()
    {
    }
  };

  TestSuite* test = new TestSuite(argc, argv);

  test->add(HDF5DataSetTest::suite());
  test->add(OGRDataSetDriverTest::suite());
  test->add(HDF5DataSetDriverTest::suite());

  return test;
}

