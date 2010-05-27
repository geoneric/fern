#ifndef INCLUDED_UNARYSAMETEST
#include "UnarySameTest.h"
#define INCLUDED_UNARYSAMETEST
#endif

#ifndef INCLUDED_BOOST_SHARED_PTR
#include <boost/shared_ptr.hpp>
#define INCLUDED_BOOST_SHARED_PTR
#endif

#ifndef INCLUDED_BOOST_TEST_TEST_TOOLS
#include <boost/test/test_tools.hpp>
#define INCLUDED_BOOST_TEST_TEST_TOOLS
#endif

#ifndef INCLUDED_BOOST_TEST_UNIT_TEST_SUITE
#include <boost/test/unit_test_suite.hpp>
#define INCLUDED_BOOST_TEST_UNIT_TEST_SUITE
#endif

#ifndef INCLUDED_RANALLY_OPERATIONS_LOCAL_UNARY_FRAMEWORK
#include "Ranally/Operations/Local/Unary/Framework.h"
#define INCLUDED_RANALLY_OPERATIONS_LOCAL_UNARY_FRAMEWORK
#endif

#ifndef INCLUDED_RANALLY_OPERATIONS_LOCAL_UNARY_PLUS
#include "Ranally/Operations/Local/Unary/Plus.h"
#define INCLUDED_RANALLY_OPERATIONS_LOCAL_UNARY_PLUS
#endif

#ifndef INCLUDED_RANALLY_OPERATIONS_POLICIES_NO_DATA
#include "Ranally/Operations/Policies/NoData.h"
#define INCLUDED_RANALLY_OPERATIONS_POLICIES_NO_DATA
#endif



//------------------------------------------------------------------------------
// DEFINITION OF STATIC UNARYSAMETEST MEMBERS
//------------------------------------------------------------------------------

boost::unit_test::test_suite* UnarySameTest::suite()
{
  boost::unit_test::test_suite* suite = BOOST_TEST_SUITE(__FILE__);
  boost::shared_ptr<UnarySameTest> instance(
         new UnarySameTest());
  suite->add(BOOST_CLASS_TEST_CASE(
         &UnarySameTest::test, instance));

  return suite;
}



//------------------------------------------------------------------------------
// DEFINITION OF UNARYSAMETEST MEMBERS
//------------------------------------------------------------------------------

//! Constructor.
UnarySameTest::UnarySameTest()
{
}



void UnarySameTest::test()
{
  using namespace ranally::operations::unary;
  using namespace ranally::operations::policies;

  // // TODO Use operation with an input domain.

  {
    framework::UnarySame<int, bool, plus::Algorithm, plus::DomainPolicy,
         plus::RangePolicy, TestNoDataValue> operation;
    int result;
    bool isNoData;

  //   result = -999;
  //   isNoData = false;
  //   operation(3, 4, result, isNoData);
  //   BOOST_CHECK(!isNoData);
  //   BOOST_CHECK_EQUAL(result, 7);

  //   // No-data input.
  //   result = -999;
  //   isNoData = true;
  //   operation(3, 4, result, isNoData);
  //   BOOST_CHECK(isNoData);
  //   BOOST_CHECK_EQUAL(result, -999);

  //   // TODO Argument out of domain.

  //   // Result out of range.
  //   result = -999;
  //   isNoData = false;
  //   operation(std::numeric_limits<int>::max(), 1, result, isNoData);
  //   BOOST_CHECK(isNoData);
  }

  // TODO Test overloads.
}

