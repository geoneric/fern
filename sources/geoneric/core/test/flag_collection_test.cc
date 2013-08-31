#define BOOST_TEST_MODULE geoneric core
#include <boost/test/unit_test.hpp>
#include "geoneric/core/flag_collection.h"


enum MyFlags {
    MF_YES,
    MF_NO,
    MF_MAYBE,
    MF_NR_MY_FLAGS
};


class MyFlagCollection:
    public geoneric::FlagCollection<MyFlagCollection, MyFlags, MF_NR_MY_FLAGS>
{

    typedef geoneric::FlagCollection<MyFlagCollection, MyFlags, MF_NR_MY_FLAGS>
        Base;

public:

    MyFlagCollection()=default;

    MyFlagCollection(unsigned long long flags)
        : Base(flags)
    {
    }

    geoneric::String to_string() const { return "blah"; }

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
