#ifndef INCLUDED_BINARYSAMETEST
#include "BinarySameTest.h"
#define INCLUDED_BINARYSAMETEST
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

#ifndef INCLUDED_RANALLY_OPERATIONS_LOCAL_BINARY_FRAMEWORK
#include "Ranally/Operations/Local/Binary/Framework.h"
#define INCLUDED_RANALLY_OPERATIONS_LOCAL_BINARY_FRAMEWORK
#endif

#ifndef INCLUDED_RANALLY_OPERATIONS_LOCAL_BINARY_PLUS
#include "Ranally/Operations/Local/Binary/Plus.h"
#define INCLUDED_RANALLY_OPERATIONS_LOCAL_BINARY_PLUS
#endif

#ifndef INCLUDED_RANALLY_OPERATIONS_POLICIES_CLIPMASK
#include "Ranally/Operations/Policies/ClipMask.h"
#define INCLUDED_RANALLY_OPERATIONS_POLICIES_CLIPMASK
#endif

#ifndef INCLUDED_RANALLY_OPERATIONS_POLICIES_NODATA
#include "Ranally/Operations/Policies/NoData.h"
#define INCLUDED_RANALLY_OPERATIONS_POLICIES_NODATA
#endif



//------------------------------------------------------------------------------
// DEFINITION OF STATIC BINARYSAMETEST MEMBERS
//------------------------------------------------------------------------------

boost::unit_test::test_suite* BinarySameTest::suite()
{
  boost::unit_test::test_suite* suite = BOOST_TEST_SUITE(__FILE__);
  boost::shared_ptr<BinarySameTest> instance(
         new BinarySameTest());
  suite->add(BOOST_CLASS_TEST_CASE(
         &BinarySameTest::test, instance));

  return suite;
}



//------------------------------------------------------------------------------
// DEFINITION OF BINARYSAMETEST MEMBERS
//------------------------------------------------------------------------------

//! Constructor.
BinarySameTest::BinarySameTest()
{
}



void BinarySameTest::test()
{
  using namespace ranally::operations::binary;
  using namespace ranally::operations::policies;

  // TODO Use operation with an input domain.

  {
    framework::BinarySame<int, bool, bool, plus::Algorithm, plus::DomainPolicy,
         plus::RangePolicy, TestClipMaskValue, TestNoDataValue> operation;
    int result;
    bool clipMask, isNoData;

    result = -999;
    clipMask = false;
    isNoData = false;
    operation(3, 4, result, clipMask, isNoData);
    BOOST_CHECK(!isNoData);
    BOOST_CHECK_EQUAL(result, 7);

    // Clip mask.
    result = -999;
    clipMask = true;
    isNoData = false;
    operation(3, 4, result, clipMask, isNoData);
    BOOST_CHECK(!isNoData);
    BOOST_CHECK_EQUAL(result, -999);

    // No-data input.
    result = -999;
    clipMask = false;
    isNoData = true;
    operation(3, 4, result, clipMask, isNoData);
    BOOST_CHECK(isNoData);
    BOOST_CHECK_EQUAL(result, -999);

    // TODO Argument out of domain.

    // Result out of range.
    result = -999;
    clipMask = false;
    isNoData = false;
    operation(std::numeric_limits<int>::max(), 1, result, clipMask, isNoData);
    BOOST_CHECK(isNoData);
  }

  // TODO Test overloads.
}
