#define BOOST_TEST_MODULE fern algorithm policy
#include <boost/test/unit_test.hpp>
#include "fern/algorithm/policy/discard_domain_errors.h"


namespace fa = fern::algorithm;


BOOST_AUTO_TEST_SUITE(discard_no_data)

BOOST_AUTO_TEST_CASE(test)
{
    {
        fa::DiscardDomainErrors<> policy;
        BOOST_CHECK(policy.within_domain());
    }

    {
        fa::DiscardDomainErrors<int32_t> policy;
        BOOST_CHECK(policy.within_domain(5));
    }

    {
        fa::DiscardDomainErrors<int32_t, int32_t> policy;
        BOOST_CHECK(policy.within_domain(5, 6));
    }
}

BOOST_AUTO_TEST_SUITE_END()
