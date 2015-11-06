// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern algorithm policy discard_domain_errors
#include <boost/test/unit_test.hpp>
#include "fern/algorithm/policy/discard_domain_errors.h"


namespace fa = fern::algorithm;


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
