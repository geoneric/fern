#pragma once
#include <memory>
#include "fern/core/thread_pool.h"


namespace fern {
namespace algorithm {

//! Execution policy class for parallel execution of algorithms.
/*!
    @ingroup    fern_algorithm_policy_group
    @sa         sequential, parallel, ExecutionPolicy
*/
class ParallelExecutionPolicy
{

public:

                   ParallelExecutionPolicy(
                                        size_t nr_threads=
                                            hardware_concurrency());

                   ParallelExecutionPolicy(
                                        ParallelExecutionPolicy const& other)
                                            =default;

                   ParallelExecutionPolicy(
                                        ParallelExecutionPolicy&& other)
                                            =default;

                   ~ParallelExecutionPolicy()=default;

    ParallelExecutionPolicy&
                   operator=           (ParallelExecutionPolicy const& other)
                                            =default;

    ParallelExecutionPolicy&
                   operator=           (ParallelExecutionPolicy&& other)
                                            =default;

    ThreadPool&    thread_pool         ();

private:

    std::shared_ptr<ThreadPool> _thread_pool;

};

} // namespace algorithm
} // namespace fern
