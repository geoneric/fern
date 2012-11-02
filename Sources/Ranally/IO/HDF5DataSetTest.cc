#include "Ranally/IO/HDF5DataSetTest.h"
#include <boost/test/test_tools.hpp>
#include <boost/test/unit_test_suite.hpp>


boost::unit_test::test_suite* HDF5DataSetTest::suite()
{
    boost::unit_test::test_suite* suite = BOOST_TEST_SUITE(__FILE__);
    boost::shared_ptr<HDF5DataSetTest> instance(
        new HDF5DataSetTest());
    suite->add(BOOST_CLASS_TEST_CASE(
        &HDF5DataSetTest::test, instance));

    return suite;
}


HDF5DataSetTest::HDF5DataSetTest()
{
}


void HDF5DataSetTest::test()
{
    bool testImplemented = false;
    BOOST_WARN(testImplemented);
}
