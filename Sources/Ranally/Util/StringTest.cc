#include "Ranally/Util/StringTest.h"
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>
#include "Ranally/Util/String.h"


boost::unit_test::test_suite* StringTest::suite()
{
    boost::unit_test::test_suite* suite = BOOST_TEST_SUITE(__FILE__);
    boost::shared_ptr<StringTest> instance(new StringTest());
    suite->add(BOOST_CLASS_TEST_CASE(
        &StringTest::testEncodeInUTF8, instance));

    return suite;
}


StringTest::StringTest()
{
}


void StringTest::testEncodeInUTF8()
{
}
