// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#pragma once
#include <memory>
#include "fern/core/thread_pool.h"


namespace fern {
namespace algorithm {

//! Execution policy class for parallel execution of algorithms.
/*!
    @ingroup    fern_algorithm_policy_group
    @sa         SequentialExecutionPolicy, ExecutionPolicy
*/
class ParallelExecutionPolicy
{

public:

                   ParallelExecutionPolicy(
                                        std::size_t nr_threads=
                                            hardware_concurrency());

                   ParallelExecutionPolicy(
                                        ParallelExecutionPolicy const&)=default;

                   ParallelExecutionPolicy(
                                        ParallelExecutionPolicy&&)=default;

                   ~ParallelExecutionPolicy()=default;

    ParallelExecutionPolicy&
                   operator=           (ParallelExecutionPolicy const&)=default;

    ParallelExecutionPolicy&
                   operator=           (ParallelExecutionPolicy&&)=default;

    ThreadPool&    thread_pool         ();

private:

    std::shared_ptr<ThreadPool> _thread_pool;

};

} // namespace algorithm
} // namespace fern
