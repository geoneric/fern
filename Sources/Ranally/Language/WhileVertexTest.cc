#include "Ranally/Language/WhileVertexTest.h"
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>


boost::unit_test::test_suite* WhileVertexTest::suite()
{
    boost::unit_test::test_suite* suite = BOOST_TEST_SUITE(__FILE__);
    boost::shared_ptr<WhileVertexTest> instance(
        new WhileVertexTest());
    suite->add(BOOST_CLASS_TEST_CASE(
        &WhileVertexTest::test, instance));

    return suite;
}


WhileVertexTest::WhileVertexTest()
{
}


void WhileVertexTest::test()
{
    bool testImplemented = false;
    BOOST_WARN(testImplemented);
}
