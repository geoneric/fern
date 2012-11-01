#include "Ranally/Language/ScriptVertexTest.h"
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>


boost::unit_test::test_suite* ScriptVertexTest::suite()
{
    boost::unit_test::test_suite* suite = BOOST_TEST_SUITE(__FILE__);
    boost::shared_ptr<ScriptVertexTest> instance(
        new ScriptVertexTest());
    suite->add(BOOST_CLASS_TEST_CASE(
        &ScriptVertexTest::test, instance));

    return suite;
}


ScriptVertexTest::ScriptVertexTest()
{
}


void ScriptVertexTest::test()
{
    bool testImplemented = false;
    BOOST_WARN(testImplemented);
}
