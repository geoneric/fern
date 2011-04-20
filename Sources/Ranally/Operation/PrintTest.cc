#include "PrintTest.h"

#include <sstream>
#include <boost/shared_ptr.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>
#include "Ranally/Raster.h"
#include "Ranally/Operation/Print.h"



boost::unit_test::test_suite* PrintTest::suite()
{
  boost::unit_test::test_suite* suite = BOOST_TEST_SUITE(__FILE__);
  boost::shared_ptr<PrintTest> instance(
    new PrintTest());
  suite->add(BOOST_CLASS_TEST_CASE(
    &PrintTest::test, instance));

  return suite;
}



PrintTest::PrintTest()
{
}



void PrintTest::test()
{
  std::stringstream stream;

  {
    int scalar = 5;
    ranally::operation::print(scalar, stream);
    BOOST_CHECK_EQUAL(stream.str(), "5\n");
  }

  {
    int array[10] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    ranally::operation::print(array, stream);
    BOOST_CHECK_EQUAL(stream.str(), "[1, 2, 3, ..., 7, 8, 9, 10]\n");
  }

  {
    ranally::Raster<int, 20, 30> raster;
    for(size_t r = 0; r < raster.nrRows(); ++r) {
      for(size_t c = 0; c < raster.nrCols(); ++c) {
        raster.set(r, c, r * raster.nrCols() + c);
      }
    }

    ranally::operation::print(raster, stream);
    BOOST_CHECK_EQUAL(stream.str(),
      "[[ 1,  2,  3, ..., 28, 29,  30]\n"
      " [31, 32, 33, ..., 58, 59,  60]\n"
      " [61, 62, 63, ..., 88, 89,  90]\n"
      " ...\n"
      " [61, 62, 63, ..., 88, 89, 600]\n"
    );
  }

}

