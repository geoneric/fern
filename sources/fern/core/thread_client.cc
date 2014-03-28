#include "fern/core/thread_client.h"
#include <cassert>
#include "fern/core/memory.h"


namespace fern {

static std::unique_ptr<ThreadPool> thread_pool;


ThreadPool& ThreadClient::pool()
{
    // You did instantiate a ThreadClient instance, didn't you?
    assert(thread_pool);

    return *thread_pool;
}


ThreadClient::ThreadClient(
    size_t nr_threads)
{
    // Instantiate the thread pool only once.
    assert(!thread_pool);
    thread_pool = std::make_unique<ThreadPool>(nr_threads);
}


ThreadClient::~ThreadClient()
{
    assert(thread_pool);
    thread_pool.reset();
}

} // namespace fern
