#include "fern/algorithm/policy/parallel_execution_policy.h"


namespace fern {
namespace algorithm {

ParallelExecutionPolicy::ParallelExecutionPolicy(
    size_t nr_threads)

    : _thread_pool(std::make_shared<ThreadPool>(nr_threads))

{
}


ThreadPool& ParallelExecutionPolicy::thread_pool()
{
    return *_thread_pool;
}

} // namespace algorithm
} // namespace fern
