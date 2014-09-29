#pragma once
#include <boost/variant.hpp>


namespace fern {
namespace algorithm {

//! Execution policy class for sequential execution of algorithms.
/*!
    \sa        sequential, parallel, ExecutionPolicy
*/
class SequentialExecutionPolicy{};


//! Execution policy class for parallel execution of algorithms.
/*!
    \sa        sequential, parallel, ExecutionPolicy
*/
class ParallelExecutionPolicy{};


//! Execution policy instance for sequential execution of algorithms.
/*!
    \sa        parallel, ExecutionPolicy
*/
constexpr SequentialExecutionPolicy sequential = SequentialExecutionPolicy();


//! Execution policy instance for parallel execution of algorithms.
/*!
    \sa        sequential, ExecutionPolicy

    Parallel algorithms make use of the ThreadClient::pool(). Therefore,
    a ThreadClient instance must be created before calling algorithms with
    the parallel execution policy.
*/
constexpr ParallelExecutionPolicy parallel = ParallelExecutionPolicy();


//! Generic execution policy class.
/*!
    \sa        sequential, parallel

    An ExecutionPolicy instance can be created and assigned from sequential
    or parallel.

    \code
    ExecutionPolicy execution_policy = sequential;

    if(concurrent) {
        execution_policy = parallel;
    }

    my_algorithm(execution_policy, value1, value2, result);
    \endcode
*/
using ExecutionPolicy = boost::variant<SequentialExecutionPolicy,
    ParallelExecutionPolicy>;


namespace detail {

// These id's *must* correspond with the order of the types in the
// ExecutionPolicy variant. policy.which() returns these id's.
size_t const sequential_execution_policy_id = 0;
size_t const parallel_execution_policy_id = 1;


template<
    class ConcreteExecutionPolicy>
inline ConcreteExecutionPolicy& get_policy(
    ExecutionPolicy& policy)
{
    // For some reason calling boost::get with instance instead of pointer
    // conflicts with fern::get.
    return *boost::get<ConcreteExecutionPolicy>(&policy);
}


template<
    class ConcreteExecutionPolicy>
inline ConcreteExecutionPolicy const& get_policy(
    ExecutionPolicy const& policy)
{
    // For some reason calling boost::get with instance instead of pointer
    // conflicts with fern::get.
    return *boost::get<ConcreteExecutionPolicy>(&policy);
}

} // namespace detail
} // namespace algorithm
} // namespace fern
