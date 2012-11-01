#include "Ranally/Language/FunctionVertexTest.h"
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>


boost::unit_test::test_suite* FunctionVertexTest::suite()
{
    boost::unit_test::test_suite* suite = BOOST_TEST_SUITE(__FILE__);
    boost::shared_ptr<FunctionVertexTest> instance(
        new FunctionVertexTest());
    suite->add(BOOST_CLASS_TEST_CASE(
        &FunctionVertexTest::test, instance));

    return suite;
}


FunctionVertexTest::FunctionVertexTest()
{
}


void FunctionVertexTest::test()
{
    bool testImplemented = false;
    BOOST_WARN(testImplemented);
}
