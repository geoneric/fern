#define BOOST_TEST_MODULE fern algorithm policy
#include <boost/test/unit_test.hpp>
#include "fern/algorithm/policy/parallel_execution_policy.h"


BOOST_AUTO_TEST_SUITE(parallel_execution_policy)

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

BOOST_AUTO_TEST_SUITE_END()
