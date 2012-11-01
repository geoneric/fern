#include "Ranally/Language/AssignmentVertexTest.h"
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>


boost::unit_test::test_suite* AssignmentVertexTest::suite()
{
    boost::unit_test::test_suite* suite = BOOST_TEST_SUITE(__FILE__);
    boost::shared_ptr<AssignmentVertexTest> instance(
        new AssignmentVertexTest());
    suite->add(BOOST_CLASS_TEST_CASE(
        &AssignmentVertexTest::test, instance));

    return suite;
}


AssignmentVertexTest::AssignmentVertexTest()
{
}


void AssignmentVertexTest::test()
{
    bool testImplemented = false;
    BOOST_WARN(testImplemented);
}
