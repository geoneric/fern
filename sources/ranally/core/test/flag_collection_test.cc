#define BOOST_TEST_MODULE ranally core
#include <boost/test/unit_test.hpp>
#include "ranally/core/flag_collection.h"


enum MyFlags {
    MF_YES,
    MF_NO,
    MF_MAYBE,
    MF_NR_MY_FLAGS
};


class MyFlagCollection:
    public ranally::FlagCollection<MyFlagCollection, MyFlags, MF_NR_MY_FLAGS>
{

    typedef ranally::FlagCollection<MyFlagCollection, MyFlags, MF_NR_MY_FLAGS>
        Base;

public:

    MyFlagCollection()=default;

    MyFlagCollection(unsigned long long flags)
        : Base(flags)
    {
    }

    ranally::String to_string() const { return "blah"; }

};


BOOST_AUTO_TEST_SUITE(flag_collection)

BOOST_AUTO_TEST_CASE(flag_collection)
{
    MyFlagCollection flags;

    BOOST_CHECK_EQUAL(flags.count(), 0u);
    flags |= MyFlagCollection(1 << MF_YES);

    BOOST_CHECK_EQUAL(flags.count(), 1u);
    BOOST_CHECK(flags.fixed());
    BOOST_CHECK(flags.test(MF_YES));
    BOOST_CHECK(!flags.test(MF_NO));

    {
        MyFlagCollection copy(flags);
        BOOST_CHECK_EQUAL(flags.count(), 1u);
        BOOST_CHECK(flags.test(MF_YES));
    }

    {
        MyFlagCollection copy = flags;
        BOOST_CHECK_EQUAL(flags.count(), 1u);
        BOOST_CHECK(flags.test(MF_YES));
    }
}

BOOST_AUTO_TEST_SUITE_END()
