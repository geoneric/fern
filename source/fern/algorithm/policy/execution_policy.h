#pragma once
#include <boost/variant.hpp>
#include "fern/algorithm/policy/parallel_execution_policy.h"


namespace fern {
namespace algorithm {

//! Execution policy class for sequential execution of algorithms.
/*!
    @ingroup    fern_algorithm_policy_group
    @sa         sequential, parallel, ExecutionPolicy
*/
class SequentialExecutionPolicy{};


//! Execution policy instance for sequential execution of algorithms.
/*!
    @ingroup    fern_algorithm_policy_group
    @sa         parallel, ExecutionPolicy
*/
extern SequentialExecutionPolicy sequential;


//! Execution policy instance for parallel execution of algorithms.
/*!
    @ingroup    fern_algorithm_policy_group
    @sa         sequential, ExecutionPolicy
*/
extern ParallelExecutionPolicy parallel;


//! Generic execution policy class.
/*!
    @ingroup    fern_algorithm_policy_group
    @sa         sequential, parallel

    An ExecutionPolicy instance can be created and assigned from sequential
    or parallel.

    @code
    ExecutionPolicy execution_policy = sequential;

    if(concurrent) {
        execution_policy = parallel;
    }

    my_algorithm(execution_policy, value1, value2, result);
    @endcode
*/
using ExecutionPolicy = boost::variant<SequentialExecutionPolicy,
    ParallelExecutionPolicy>;


namespace detail {

// These id's *must* correspond with the order of the types in the
// ExecutionPolicy variant. policy.which() returns these id's.
size_t const sequential_execution_policy_id = 0;
size_t const parallel_execution_policy_id = 1;

} // namespace detail
} // namespace algorithm
} // namespace fern
