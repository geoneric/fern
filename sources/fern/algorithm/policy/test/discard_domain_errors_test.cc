#define BOOST_TEST_MODULE fern algorithm policy
#include <boost/test/unit_test.hpp>
#include "fern/algorithm/policy/discard_domain_errors.h"


BOOST_AUTO_TEST_SUITE(discard_no_data)

BOOST_AUTO_TEST_CASE(test)
{
    {
        fern::DiscardDomainErrors<> policy;
        BOOST_CHECK(policy.within_domain());
    }

    {
        fern::DiscardDomainErrors<int> policy;
        BOOST_CHECK(policy.within_domain(5));
    }

    {
        fern::DiscardDomainErrors<int, int> policy;
        BOOST_CHECK(policy.within_domain(5, 6));
    }
}

BOOST_AUTO_TEST_SUITE_END()
