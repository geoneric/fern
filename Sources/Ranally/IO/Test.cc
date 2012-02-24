#include <boost/test/included/unit_test.hpp>
#include "Ranally/IO/HDF5Client.h"
#include "Ranally/IO/HDF5DataSetDriverTest.h"
#include "Ranally/IO/HDF5DataSetTest.h"



boost::unit_test::test_suite* init_unit_test_suite(
         int argc,
         char** const argv) {

  struct TestSuite:
    public boost::unit_test::test_suite,
    ranally::io::HDF5Client
  {
    TestSuite(
         int& /* argc */,
         char** /* argv */)
      : boost::unit_test::test_suite("Master test suite"),
        HDF5Client()
    {
    }
  };

  TestSuite* test = new TestSuite(argc, argv);

  test->add(HDF5DataSetTest::suite());
  test->add(HDF5DataSetDriverTest::suite());

  return test;
}

