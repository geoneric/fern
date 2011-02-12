#ifndef INCLUDED_EXAMPLECODETEST
#include "ExampleCodeTest.h"
#define INCLUDED_EXAMPLECODETEST
#endif

// External headers.
#ifndef INCLUDED_BOOST_SHARED_PTR
#include <boost/shared_ptr.hpp>
#define INCLUDED_BOOST_SHARED_PTR
#endif

#ifndef INCLUDED_BOOST_TEST_FLOATING_POINT_COMPARISON
#include <boost/test/floating_point_comparison.hpp>
#define INCLUDED_BOOST_TEST_FLOATING_POINT_COMPARISON
#endif

#ifndef INCLUDED_BOOST_TEST_TEST_TOOLS
#include <boost/test/test_tools.hpp>
#define INCLUDED_BOOST_TEST_TEST_TOOLS
#endif

#ifndef INCLUDED_BOOST_TEST_UNIT_TEST_SUITE
#include <boost/test/unit_test_suite.hpp>
#define INCLUDED_BOOST_TEST_UNIT_TEST_SUITE
#endif

// Project headers.

// Module headers.
#ifndef INCLUDED_EXAMPLECODE
#include "ExampleCode.h"
#define INCLUDED_EXAMPLECODE
#endif



/*!
  \file
  This file contains the implementation of the ExampleCodeTest class.
*/



// namespace {

//------------------------------------------------------------------------------
// DEFINITION OF STATIC EXAMPLECODETEST MEMBERS
//------------------------------------------------------------------------------

//! Suite.
boost::unit_test::test_suite* ExampleCodeTest::suite()
{
  boost::unit_test::test_suite* suite = BOOST_TEST_SUITE(__FILE__);
  boost::shared_ptr<ExampleCodeTest> instance(
         new ExampleCodeTest());
  suite->add(BOOST_CLASS_TEST_CASE(
         &ExampleCodeTest::testCollectionCollection, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
         &ExampleCodeTest::testCollectionConstant, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
         &ExampleCodeTest::testMask, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
         &ExampleCodeTest::testSqrt, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
         &ExampleCodeTest::testAbs, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
         &ExampleCodeTest::testGreaterThan, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
         &ExampleCodeTest::testOverloads, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
         &ExampleCodeTest::testNAryOperation, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
         &ExampleCodeTest::testIsNull, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
         &ExampleCodeTest::testSetNull, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
         &ExampleCodeTest::testCollectorOperation, instance));
  suite->add(BOOST_CLASS_TEST_CASE(
         &ExampleCodeTest::testIgnoreNoData, instance));

  return suite;
}



//------------------------------------------------------------------------------
// DEFINITION OF EXAMPLECODETEST MEMBERS
//------------------------------------------------------------------------------

//! Constructor.
ExampleCodeTest::ExampleCodeTest()
{
}



void ExampleCodeTest::init()
{
  _intCollection1.clear();
  _intCollection1.push_back(1);
  _intCollection1.push_back(2);
  _intCollection1.push_back(0);
  _intCollection1.push_back(4);
  _intCollection1.push_back(5);

  _intCollection2.clear();
  _intCollection2.push_back(10);
  _intCollection2.push_back(20);
  _intCollection2.push_back(30);
  _intCollection2.push_back(40);
  _intCollection2.push_back(0);

  _intConstant = 5;

  _floatCollection1.clear();
  _floatCollection1.push_back(25.0);
  _floatCollection1.push_back(9.0);
  _floatCollection1.push_back(-49.0);

  _floatCollection2.clear();
  _floatCollection2.push_back(24.0);
  _floatCollection2.push_back(10.0);
  _floatCollection2.push_back(-50.0);

  _intResult.clear();
  _floatResult.clear();
  _boolResult.clear();
}



void ExampleCodeTest::testConstructor()
{
  /// UnaryMinus<int> unaryMinus;
  // BinaryPlus<int> binaryPlus;

  /// // No additional memory is required to perform this operation.
  /// BOOST_CHECK_EQUAL(unaryMinus.memoryRequirements, size_t(0));
  // BOOST_CHECK_EQUAL(BinaryPlus<int>::memoryRequirements, size_t(0));
}



void ExampleCodeTest::testCollectionCollection()
{
  UnaryMinus<int> unaryMinus;
  BinaryPlus<int> binaryPlus;

  // result = -collection1
  {
    init();
    _intResult.resize(_intCollection1.size(), 99);
    unaryMinus(_intResult.begin(), _intCollection1.begin(), _intResult.size());

    BOOST_CHECK_EQUAL(_intResult[0], -1);
    BOOST_CHECK_EQUAL(_intResult[1], -2);
    BOOST_CHECK_EQUAL(_intResult[2], 0);
    BOOST_CHECK_EQUAL(_intResult[3], -4);
    BOOST_CHECK_EQUAL(_intResult[4], -5);
  }

  // result = collection1 + collection2
  {
    init();
    _intResult.resize(_intCollection1.size(), 99);
    binaryPlus(_intResult.begin(), _intCollection1.begin(),
         _intCollection2.begin(), _intResult.size());

    BOOST_CHECK_EQUAL(_intResult[0], 11);
    BOOST_CHECK_EQUAL(_intResult[1], 22);
    BOOST_CHECK_EQUAL(_intResult[2], 30);
    BOOST_CHECK_EQUAL(_intResult[3], 44);
    BOOST_CHECK_EQUAL(_intResult[4], 5);
  }

  // collection1 += collection2
  {
    init();
    binaryPlus(_intCollection1.begin(), _intCollection1.begin(),
         _intCollection2.begin(), _intCollection1.size());

    BOOST_CHECK_EQUAL(_intCollection1[0], 11);
    BOOST_CHECK_EQUAL(_intCollection1[1], 22);
    BOOST_CHECK_EQUAL(_intCollection1[2], 30);
    BOOST_CHECK_EQUAL(_intCollection1[3], 44);
    BOOST_CHECK_EQUAL(_intCollection1[4], 5);
  }
}



void ExampleCodeTest::testCollectionConstant()
{
  UnaryMinus<int> unaryMinus;
  BinaryPlus<int> binaryPlus;

  // result = -constant
  {
    init();
    _intResult.resize(_intCollection1.size(), 99);
    unaryMinus(_intResult.begin(), _intConstant, _intResult.size());

    BOOST_CHECK_EQUAL(_intResult[0], -5);
    BOOST_CHECK_EQUAL(_intResult[1], -5);
    BOOST_CHECK_EQUAL(_intResult[2], -5);
    BOOST_CHECK_EQUAL(_intResult[3], -5);
    BOOST_CHECK_EQUAL(_intResult[4], -5);
  }

  // result = collection + constant
  {
    init();
    _intResult.resize(_intCollection1.size(), 99);
    binaryPlus(_intResult.begin(), _intCollection1.begin(), _intConstant,
         _intResult.size());

    BOOST_CHECK_EQUAL(_intResult[0], 6);
    BOOST_CHECK_EQUAL(_intResult[1], 7);
    BOOST_CHECK_EQUAL(_intResult[2], 5);
    BOOST_CHECK_EQUAL(_intResult[3], 9);
    BOOST_CHECK_EQUAL(_intResult[4], 10);
  }

  // collection += constant
  {
    init();
    _intResult.resize(_intCollection1.size(), 99);
    binaryPlus(_intCollection1.begin(), _intCollection1.begin(), _intConstant,
         _intResult.size());

    BOOST_CHECK_EQUAL(_intCollection1[0], 6);
    BOOST_CHECK_EQUAL(_intCollection1[1], 7);
    BOOST_CHECK_EQUAL(_intCollection1[2], 5);
    BOOST_CHECK_EQUAL(_intCollection1[3], 9);
    BOOST_CHECK_EQUAL(_intCollection1[4], 10);
  }
}



void ExampleCodeTest::testMask()
{
  BinaryPlus<int, int,
         IgnoreNoData<int>,
         DontCheckValueDomain<int>,
         MaskByValuePredicate< std::vector<int>::const_iterator > > binaryPlus;
  binaryPlus.setMaskPredicate(std::bind2nd(std::equal_to<int>(), 0));
  binaryPlus.setValues(_intCollection1.begin(), _intCollection2.begin());

  // result = collection1 + collection2
  // Skip those values that equal 0.
  {
    init();
    _intResult.resize(_intCollection1.size(), 99);
    binaryPlus(_intResult.begin(), _intCollection1.begin(),
         _intCollection2.begin(), _intResult.size());

    // _intResult is initialized with 99's.
    BOOST_CHECK_EQUAL(_intResult[0], 11);
    BOOST_CHECK_EQUAL(_intResult[1], 22);
    BOOST_CHECK_EQUAL(_intResult[2], 99);
    BOOST_CHECK_EQUAL(_intResult[3], 44);
    BOOST_CHECK_EQUAL(_intResult[4], 99);
  }
}



void ExampleCodeTest::testSqrt()
{
  // // result = sqrt(collection1)
  // {
  //   SquareRoot<int> squareRoot;
  //   init();

  //   // Yahoo! Does not compile because the template argument is non-float.
  //   squareRoot(_intResult.begin(), _intCollection1.begin(), _intResult.size());
  // }

  // result = sqrt(collection1)
  SquareRoot<float> squareRoot;

  {
    init();
    _floatResult.resize(_floatCollection1.size());
    squareRoot(_floatResult.begin(), _floatCollection1.begin(),
         _floatResult.size());

    BOOST_CHECK_EQUAL(_floatResult[0], 5.0f);
    BOOST_CHECK_EQUAL(_floatResult[1], 3.0f);
    BOOST_CHECK(std::isnan(_floatResult[2]));
  }

  // Square root of collection with a negative value. Prove that by setting
  // a value domain, negative values can be skipped (zero is also skipped by
  // the predicate).
  {
    SquareRoot<float, float,
           IgnoreNoData<float>,
           DomainByPredicate<float> > squareRoot;
    squareRoot.setDomainPredicate(std::bind2nd(std::greater<float>(), 0.0f));
    init();
    _floatResult.resize(_floatCollection1.size(), -999.0f);
    squareRoot(_floatResult.begin(), _floatCollection1.begin(),
         _floatResult.size());
    BOOST_CHECK_EQUAL(_floatResult[0], 5.0f);
    BOOST_CHECK_EQUAL(_floatResult[1], 3.0f);
    BOOST_CHECK_EQUAL(_floatResult[2], -999.0f);
  }
}



void ExampleCodeTest::testAbs()
{
  {
    AbsOperation<int> operation;
    int result;

    operation(result, -3);
    BOOST_CHECK_EQUAL(result, 3);
  }

  {
    AbsOperation<double> operation;
    double result;

    operation(result, -3.3);
    BOOST_CHECK_CLOSE(result, 3.3, 0.001);
  }

  // OK: compile error, ambiguous.
  // {
  //   AbsOperation<size_t> operation;
  //   size_t result;

  //   operation(result, 3);
  //   BOOST_CHECK_EQUAL(result, 3);
  // }
}



void ExampleCodeTest::testGreaterThan()
{
  {
    // result = collection1 > collection2
    GreaterThan<float, bool> greaterThan;

    init();
    _boolResult.resize(_floatCollection1.size());
    greaterThan(_boolResult.begin(), _floatCollection1.begin(),
         _floatCollection2.begin(), _boolResult.size());

    BOOST_CHECK_EQUAL(_boolResult[0], true);
    BOOST_CHECK_EQUAL(_boolResult[1], false);
    BOOST_CHECK_EQUAL(_boolResult[2], true);
  }

  {
    // result = collection1 > collection2
    GreaterThan<float, int> greaterThan;

    init();
    _intResult.resize(_floatCollection1.size());
    greaterThan(_intResult.begin(), _floatCollection1.begin(),
         _floatCollection2.begin(), _intResult.size());

    BOOST_CHECK_EQUAL(_intResult[0], 1);
    BOOST_CHECK_EQUAL(_intResult[1], 0);
    BOOST_CHECK_EQUAL(_intResult[2], 1);
  }
}



void ExampleCodeTest::testOverloads()
{
  BinaryMinus<int> minus;

  // result = collection1 - collection2
  {
    init();
    _intResult.resize(_intCollection1.size(), 99);
    minus(_intResult.begin(), _intCollection1.begin(), _intCollection2.begin(),
         _intResult.size());

    BOOST_CHECK_EQUAL(_intResult[0], -9);
    BOOST_CHECK_EQUAL(_intResult[1], -18);
    // ...
    BOOST_CHECK_EQUAL(_intResult[4], 5);
  }

  // result = collection1 - constant
  {
    minus(_intResult.begin(), _intCollection1.begin(), _intConstant,
         _intResult.size());

    BOOST_CHECK_EQUAL(_intResult[0], -4);
    BOOST_CHECK_EQUAL(_intResult[1], -3);
    // ...
    BOOST_CHECK_EQUAL(_intResult[4], 0);
  }

  // result = constant - collection
  {
    minus(_intResult.begin(), _intConstant, _intCollection1.begin(),
         _intResult.size());

    BOOST_CHECK_EQUAL(_intResult[0], 4);
    BOOST_CHECK_EQUAL(_intResult[1], 3);
    // ...
    BOOST_CHECK_EQUAL(_intResult[4], 0);
  }

  // result = constant - constant
  {
    int result = 99;
    minus(result, 5, 3);
    BOOST_CHECK_EQUAL(result, 2);
  }
}



void ExampleCodeTest::testNAryOperation()
{
  // result = max(collections)
  // Using std::vector collections.
  {
    Max<int> max;
    std::vector<int> collection1;
    std::vector<int> collection2;

    collection1.push_back(2); collection2.push_back(3);
    collection1.push_back(0); collection2.push_back(-1);

    std::vector< std::vector<int>::const_iterator > collections;
    collections.push_back(collection1.begin());
    collections.push_back(collection2.begin());

    std::vector<int> result(collection1.size(), 99);

    max(result.begin(), collections.begin(), collection1.size(), result.size());

    BOOST_CHECK_EQUAL(result[0], 3);
    BOOST_CHECK_EQUAL(result[1], 0);
  }

  // result = max(collections)
  // Using C-style arrays.
  {
    Max<int> max;
    int collection1[3];
    int collection2[3];

    collection1[0] = 3; collection2[0] = 5;
    collection1[1] = 1; collection2[1] = -2;
    collection1[2] = 8; collection2[2] = -5;

    int* collections[3];
    collections[0] = collection1;
    collections[1] = collection2;
    collections[2] = collection2;

    int result[2];
    result[0] = 99;
    result[1] = 99;
    result[2] = 99;

    max(result, collections, 2, 3);

    BOOST_CHECK_EQUAL(result[0], 5);
    BOOST_CHECK_EQUAL(result[1], 1);
    BOOST_CHECK_EQUAL(result[2], 8);
  }

  // result = max(collections)
  // Using C-style arrays.
  // With certain values masked out.
  {
    size_t nrValues = 3;
    size_t nrCollections = 3;
    int collection1[3];
    int collection2[3];
    int collection3[3];

    collection1[0] = 3; collection2[0] = 5;  collection3[0] = 7;
    collection1[1] = 1; collection2[1] = -2; collection3[1] = -7;
    collection1[2] = 8; collection2[2] = -5; collection3[2] = 17;

    int* collections[nrCollections];
    collections[0] = collection1;
    collections[1] = collection2;
    collections[2] = collection3;

    int result[nrValues];
    result[0] = 99;
    result[1] = 99;
    result[2] = 99;

    Max<int, int,
         IgnoreNoData<int>,
         DontCheckValueDomain<int>,
         MaskByValuePredicate<int*> > max;
    max.setMaskPredicate(std::bind2nd(std::less<int>(), 0));
    max.setValues(collections, nrCollections);

    max(result, collections, nrCollections, nrValues);

    BOOST_CHECK_EQUAL(result[0], 7);
    BOOST_CHECK_EQUAL(result[1], 99);
    BOOST_CHECK_EQUAL(result[2], 99);

  }

  {
    // Now restrict domain using a predicate.
    size_t nrValues = 4;
    int collection1[nrValues];
    int collection2[nrValues];
    int collection3[nrValues];

    collection1[0] =  3; collection2[0] =  5; collection3[0] =  7;
    collection1[1] = -1; collection2[1] = -2; collection3[1] = -7;
    collection1[2] =  8; collection2[2] =  5; collection3[2] = 17;
    collection1[3] =  8; collection2[3] = -5; collection3[3] = 17;

    size_t nrCollections = 3;
    int* collections[nrCollections];
    collections[0] = collection1;
    collections[1] = collection2;
    collections[2] = collection3;

    int result[nrValues];
    result[0] = 99;
    result[1] = 99;
    result[2] = 99;
    result[3] = 99;

    int noDataValue = -9999;

    Max<int, int,
         NoDataValue<int>,
         DomainByPredicate<int> > max;
    max.setNoDataValue(noDataValue);
    max.setDomainPredicate(std::bind2nd(std::greater_equal<int>(), 0));

    max(result, collections, nrCollections, nrValues);

    BOOST_CHECK_EQUAL(result[0], 7);
    BOOST_CHECK_EQUAL(result[1], -9999);
    BOOST_CHECK_EQUAL(result[2], 17);
    BOOST_CHECK_EQUAL(result[3], -9999);
  }

  // // result = max(value1, value2, value3, ...)
  // {
  //   Max<int> max;
  //   max(result, _intCollection1.begin(), _intCollection1.size());
  // }
}



void ExampleCodeTest::testIsNull()
{
  {
    size_t nrValues = 3;

    int values[nrValues];
    values[0] = -5;
    values[1] = -3;
    values[2] =  0;

    bool results[nrValues];
    results[0] = false;
    results[1] = false;
    results[2] = false;

    int noDataValue = -3;

    // IsNull finds cells that contain a no-data value. No-data cells can be
    // found by the value domain predicate and the new value can be set
    // by the no-data predicate.
    // The isNull algorithm is only used for those cells that do not contain
    // a no-data value. It only needs to return Result(0).
    IsNull<int, bool,
         NoDataValue<bool>,
         DomainByPredicate<int> > isNull;
    isNull.setNoDataValue(true);
    isNull.setDomainPredicate(std::bind2nd(std::not_equal_to<int>(),
         noDataValue));

    isNull(results, values, nrValues);

    BOOST_CHECK_EQUAL(results[0], false);
    BOOST_CHECK_EQUAL(results[1], true);
    BOOST_CHECK_EQUAL(results[2], false);
  }
}



void ExampleCodeTest::testSetNull()
{
  {
    size_t nrValues = 3;

    int values[nrValues];
    values[0] = -5;
    values[1] = -3;
    values[2] =  0;

    int results[nrValues];
    results[0] = -99;
    results[1] = -99;
    results[2] = -99;

    int noDataValue = -3;

    // SetNull works be setting cells with a specific value to no-data.
    // No-data handling is done by the no-data policy. No-data setting is
    // triggered for those cells that fall outside the value domain.
    // SetNull can be steered by configuring the domain to include those
    // cells that need to be set to no-data. Compare this to the where-clause
    // from the tool.
    // The setNull algorithm is a dummy function that just returns its
    // argument.
    SetNull<int, int,
         NoDataValue<int>,
         DomainByPredicate<int> > setNull;
    setNull.setNoDataValue(-9999);
    setNull.setDomainPredicate(std::bind2nd(std::not_equal_to<int>(),
         noDataValue));

    setNull(results, values, nrValues);

    BOOST_CHECK_EQUAL(results[0], -5);
    BOOST_CHECK_EQUAL(results[1], -9999);
    BOOST_CHECK_EQUAL(results[2], 0);
  }
}



void ExampleCodeTest::testCollectorOperation()
{
  {
    size_t nrValues = 2;
    size_t nrCollections = 3;

    int values1[nrValues];
    int values2[nrValues];
    int values3[nrValues];

    values1[0] = 10; values2[0] = 20; values3[0] = 30;
    values1[1] = 20; values2[1] = 30; values3[1] = 40;

    int* collections[nrCollections];
    collections[0] = values1;
    collections[1] = values2;
    collections[2] = values3;

    double results[nrValues];
    results[0] = -99.9;
    results[1] = -99.9;

    Mean<int, double> mean;

    mean(results, collections, nrCollections, nrValues);

    BOOST_CHECK_CLOSE(results[0], 20.0, 0.001);
    BOOST_CHECK_CLOSE(results[1], 30.0, 0.001);
  }
}



void ExampleCodeTest::testIgnoreNoData()
{
  /// {
  ///   size_t nrValues = 3;
  ///   int collection1[nrValues];
  ///   int collection2[nrValues];

  ///   collection1[0] =  3; collection2[0] =  5;
  ///   collection1[1] = -1; collection2[1] =  2;
  ///   collection1[2] =  8; collection2[2] = -5;

  ///   size_t nrCollections = 2;
  ///   int* collections[nrCollections];
  ///   collections[0] = collection1;
  ///   collections[1] = collection2;

  ///   int result[nrValues];
  ///   result[0] = 99;
  ///   result[1] = 99;
  ///   result[2] = 99;

  ///   int noDataValue = -9999;

  ///   Max<int, int,
  ///        NoDataValue<int>,
  ///        DomainByPredicate<int> > max;
  ///   max.setNoDataValue(noDataValue);
  ///   max.setDomainPredicate(std::bind2nd(std::greater_equal<int>(), 0));

  ///   max(result, collections, nrCollections, nrValues);

  ///   BOOST_CHECK_EQUAL(result[0], 5);
  ///   BOOST_CHECK_EQUAL(result[1], 2);
  ///   BOOST_CHECK_EQUAL(result[2], 8);
  /// }
}

// } // namespace

