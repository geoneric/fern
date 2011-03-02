#include "Ranally/Algorithm/PlusTest.h"

#include <boost/array.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>
#include "Ranally/Algorithm/Plus.h"



namespace {

template<
  typename value_type,
  size_t nr_rows,
  size_t nr_cols>
class Raster
{
private:

  boost::array<value_type, nr_rows * nr_cols> _cells;

public:

  Raster()
  {
  }

  // Raster(
  //   value_type initialValue)
  //   : _cells(initialValue)
  // {
  // }

};

} // Anonymous namespace



boost::unit_test::test_suite* PlusTest::suite()
{
  boost::unit_test::test_suite* suite = BOOST_TEST_SUITE(__FILE__);
  boost::shared_ptr<PlusTest> instance(
    new PlusTest());
  suite->add(BOOST_CLASS_TEST_CASE(
    &PlusTest::testArgumentAndResultTypes, instance));

  return suite;
}



PlusTest::PlusTest()
{
}



void PlusTest::testArgumentAndResultTypes()
{
  // int + int -> int
  {
    int argument1 = 3;
    int argument2 = 4;
    int result = 0;
    ranally::algorithm::plus(argument1, argument2, result);
    BOOST_CHECK_EQUAL(result, 7);
  }

  // int + int[2] -> int[2]
  {
    int argument1 = 3;
    int argument2[2] = {4, 5};
    int result[2] = {0, 0};
    ranally::algorithm::plus(argument1, argument2, result);
    BOOST_CHECK_EQUAL(result[0], 7);
    BOOST_CHECK_EQUAL(result[1], 8);
  }

  // int[2] + int -> int[2]
  {
    int argument1[2] = {3, 4};
    int argument2 = 4;
    int result[2] = {0, 0};
    ranally::algorithm::plus(argument1, argument2, result);
    BOOST_CHECK_EQUAL(result[0], 7);
    BOOST_CHECK_EQUAL(result[1], 8);
  }

  // int[2] + int[2] -> int[2]
  {
    int argument1[2] = {3, 4};
    int argument2[2] = {4, 5};
    int result[2] = {0, 0};
    ranally::algorithm::plus(argument1, argument2, result);
    BOOST_CHECK_EQUAL(result[0], 7);
    BOOST_CHECK_EQUAL(result[1], 9);
  }

  // int + boost::array<int, 2> -> int[2]
  {
    int argument1 = 3;
    boost::array<int, 2> argument2;
    argument2[0] = 4;
    argument2[1] = 5;
    int result[2] = {0, 0};
    ranally::algorithm::plus(argument1, argument2, result);
    BOOST_CHECK_EQUAL(result[0], 7);
    BOOST_CHECK_EQUAL(result[1], 8);
  }

  // int[2] + boost::array<int, 2> -> std::vector<int>(2, 0)
  {
    int argument1[2] = {3, 4};
    boost::array<int, 2> argument2;
    argument2[0] = 4;
    argument2[1] = 5;
    std::vector<int> result(2, 0);
    ranally::algorithm::plus(argument1, argument2, result);
    BOOST_CHECK_EQUAL(result[0], 7);
    BOOST_CHECK_EQUAL(result[1], 9);
  }

  // int[2] + int -> boost::array<int, 2>
  {
    int argument1[2] = {3, 4};
    int argument2 = 4;
    int result[2] = {0, 0};
    ranally::algorithm::plus(argument1, argument2, result);
    BOOST_CHECK_EQUAL(result[0], 7);
    BOOST_CHECK_EQUAL(result[1], 8);
  }

  // Raster + Raster -> Raster
  {
    Raster<int, 1, 2> argument1;
    // argument1.set<0, 0>(3);
    // argument1.set<0, 1>(4);
    // Raster<int, 1, 2> argument2;
    // argument2.set<0, 0>(4);
    // argument2.set<0, 1>(5);
    // Raster<int, 2, 3> result(0);
    // ranally::algorithm::plus(argument1, argument2, result);
    // BOOST_CHECK_EQUAL(result.get<0, 0>(), 7);
    // BOOST_CHECK_EQUAL(result.get<0, 1>(), 8);
  }

  // // int[1][2] + int -> int[2]
  // {
  //   int argument1[1][2];
  //   argument1[0][0] = 3;
  //   argument1[0][1] = 4;
  //   int argument2 = 4;
  //   int result[2] = {0, 0};
  //   ranally::algorithm::plus(argument1, argument2, result);
  //   // BOOST_CHECK_EQUAL(result[0], 7);
  //   // BOOST_CHECK_EQUAL(result[1], 8);
  // }
}


