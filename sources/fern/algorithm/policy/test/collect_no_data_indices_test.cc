#define BOOST_TEST_MODULE fern algorithm policy
#include <boost/test/unit_test.hpp>
#include "fern/core/type_traits.h"
#include "fern/algorithm/policy/collect_no_data_indices.h"


template<
    class Collection>
struct PolicyHost:
    public fern::CollectNoDataIndices<Collection>
{
};


BOOST_AUTO_TEST_SUITE(collect_no_data_indices)

BOOST_AUTO_TEST_CASE(collect_no_data_indices)
{
    typedef std::vector<size_t> IndexContainer;
    PolicyHost<IndexContainer> policy;

    policy.mark(0);
    policy.mark(4);
    BOOST_REQUIRE_EQUAL(policy.indices().size(), 2u);
    BOOST_CHECK_EQUAL(policy.indices()[0], 0u);
    BOOST_CHECK_EQUAL(policy.indices()[1], 4u);
}

BOOST_AUTO_TEST_SUITE_END()
