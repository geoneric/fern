// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#define BOOST_TEST_MODULE fern algorithm policy
#include <boost/test/unit_test.hpp>
#include "fern/core/type_traits.h"
#include "fern/algorithm/policy/collect_no_data_indices.h"


template<
    class Collection>
struct PolicyHost:
    public fern::algorithm::CollectNoDataIndices<Collection>
{
};


BOOST_AUTO_TEST_SUITE(collect_no_data_indices)

BOOST_AUTO_TEST_CASE(collect_no_data_indices)
{
    using IndexContainer = std::vector<size_t>;
    PolicyHost<IndexContainer> policy;

    policy.mark(0);
    policy.mark(4);
    BOOST_REQUIRE_EQUAL(policy.indices().size(), 2u);
    BOOST_CHECK_EQUAL(policy.indices()[0], 0u);
    BOOST_CHECK_EQUAL(policy.indices()[1], 4u);
}

BOOST_AUTO_TEST_SUITE_END()
