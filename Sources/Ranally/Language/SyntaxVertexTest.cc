#include "Ranally/Language/SyntaxVertexTest.h"

#include <boost/shared_ptr.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>



boost::unit_test::test_suite* SyntaxVertexTest::suite()
{
  boost::unit_test::test_suite* suite = BOOST_TEST_SUITE(__FILE__);
  boost::shared_ptr<SyntaxVertexTest> instance(
    new SyntaxVertexTest());
  suite->add(BOOST_CLASS_TEST_CASE(
    &SyntaxVertexTest::test, instance));

  return suite;
}



SyntaxVertexTest::SyntaxVertexTest()
{
}



void SyntaxVertexTest::test()
{
  bool testImplemented = false;
  BOOST_WARN(testImplemented);
}

