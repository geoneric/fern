#include "Ranally/Language/OperatorVertexTest.h"
#include <boost/shared_ptr.hpp>
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>


boost::unit_test::test_suite* OperatorVertexTest::suite()
{
    boost::unit_test::test_suite* suite = BOOST_TEST_SUITE(__FILE__);
    boost::shared_ptr<OperatorVertexTest> instance(
        new OperatorVertexTest());
    suite->add(BOOST_CLASS_TEST_CASE(
        &OperatorVertexTest::test, instance));

    return suite;
}


OperatorVertexTest::OperatorVertexTest()
{
}


void OperatorVertexTest::test()
{
    bool testImplemented = false;
    BOOST_WARN(testImplemented);
}
