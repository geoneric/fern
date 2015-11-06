// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern algorithm policy parallel_execution_policy
#include <boost/test/unit_test.hpp>
#include "fern/algorithm/policy/parallel_execution_policy.h"


BOOST_AUTO_TEST_CASE(thread_pool_size)
{
    {
        fern::algorithm::ParallelExecutionPolicy policy;

        BOOST_CHECK_EQUAL(policy.thread_pool().size(),
            fern::hardware_concurrency());
    }

    {
        fern::algorithm::ParallelExecutionPolicy policy(3);

        BOOST_CHECK_EQUAL(policy.thread_pool().size(), 3);
    }
}
