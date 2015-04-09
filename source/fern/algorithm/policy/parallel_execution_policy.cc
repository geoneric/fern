// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
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
