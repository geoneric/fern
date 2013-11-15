#define BOOST_TEST_MODULE fern core
#include <boost/test/unit_test.hpp>
#include "fern/core/flag_collection.h"


enum MyFlags {
    MF_YES,
    MF_NO,
    MF_MAYBE,
    MF_NR_MY_FLAGS
};


class MyFlagCollection:
    public fern::FlagCollection<MyFlagCollection, MyFlags, MF_NR_MY_FLAGS>
{

    typedef fern::FlagCollection<MyFlagCollection, MyFlags, MF_NR_MY_FLAGS>
        Base;

public:

    MyFlagCollection()=default;

    MyFlagCollection(unsigned long long flags)
        : Base(flags)
    {
    }

    fern::String to_string() const { return "blah"; }

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

    {
        MyFlagCollection flags1, flags2;
        flags1 |= MyFlagCollection(1 << MF_YES);
        BOOST_CHECK( (flags1 & flags2).none());
        BOOST_CHECK(!(flags1 & flags2).any());
        BOOST_CHECK(!(flags1 & flags2).fixed());

        flags2 |= MyFlagCollection(1 << MF_YES);
        BOOST_CHECK(!(flags1 & flags2).none());
        BOOST_CHECK( (flags1 & flags2).any());
        BOOST_CHECK( (flags1 & flags2).fixed());

        flags2 = MyFlagCollection(1 << MF_NO);
        BOOST_CHECK( (flags1 & flags2).none());
        BOOST_CHECK(!(flags1 & flags2).any());
        BOOST_CHECK(!(flags1 & flags2).fixed());
    }
}


BOOST_AUTO_TEST_CASE(is_subset_of)
{
    MyFlagCollection flags1, flags2;
    BOOST_CHECK(!flags1.is_subset_of(flags2));

    flags2 |= MyFlagCollection(1 << MF_YES);
    BOOST_CHECK(!flags1.is_subset_of(flags2));

    flags1 |= MyFlagCollection(1 << MF_YES);
    BOOST_CHECK(flags1.is_subset_of(flags2));

    flags1 |= MyFlagCollection(1 << MF_NO);
    BOOST_CHECK(!flags1.is_subset_of(flags2));

    flags2 |= MyFlagCollection(1 << MF_NO);
    BOOST_CHECK(flags1.is_subset_of(flags2));

    flags2 |= MyFlagCollection(1 << MF_MAYBE);
    BOOST_CHECK(flags1.is_subset_of(flags2));
}

BOOST_AUTO_TEST_SUITE_END()
