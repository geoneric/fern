// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
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

    using Base = fern::FlagCollection<MyFlagCollection, MyFlags,
        MF_NR_MY_FLAGS>;

public:

    static const MyFlagCollection YES;

    static const MyFlagCollection NO;

    static const MyFlagCollection MAYBE;

    MyFlagCollection()=default;

    MyFlagCollection(unsigned long long flags)
        : Base(flags)
    {
    }

    fern::String to_string() const { return "blah"; }

};

MyFlagCollection const MyFlagCollection::YES(1 << MF_YES);
MyFlagCollection const MyFlagCollection::NO(1 << MF_NO);
MyFlagCollection const MyFlagCollection::MAYBE(1 << MF_MAYBE);


BOOST_AUTO_TEST_SUITE(flag_collection)

BOOST_AUTO_TEST_CASE(flag_collection)
{
    BOOST_CHECK_EQUAL(MyFlagCollection::YES.count(), 1u);
    BOOST_CHECK(MyFlagCollection::YES.test(MF_YES));
    BOOST_CHECK_EQUAL(MyFlagCollection::NO.count(), 1u);
    BOOST_CHECK(MyFlagCollection::NO.test(MF_NO));
    BOOST_CHECK_EQUAL(MyFlagCollection::MAYBE.count(), 1u);
    BOOST_CHECK(MyFlagCollection::MAYBE.test(MF_MAYBE));

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
