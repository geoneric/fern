#include "fern/core/thread_client.h"
#include <cassert>
#include <iostream>
#include <memory>


namespace fern {
namespace {

static std::unique_ptr<ThreadPool> thread_pool;

} // Anonymous namespace.


//! Return the number of concurrent threads supported by the implementation.
/*!
    If this number cannot be determined reliably, this function returns 1. The
    result of this function can be used as the default size of a thread pool.
*/
size_t ThreadClient::hardware_concurrency()
{
    size_t result = std::thread::hardware_concurrency();

    return result > 0u ? result: 1u;
}


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
    // Note, that the pointer to the pool is created, but the pool itself
    // isn't. The pointer is a static variable.
    assert(!thread_pool);
    nr_threads = nr_threads > 0 ? nr_threads : 1;
    thread_pool = std::make_unique<ThreadPool>(nr_threads);
}


ThreadClient::~ThreadClient()
{
    // The pointer to the pool is a static variable.  The pool itself may or
    // may not be destructed already. This depends on whether we are static
    // or not and if so, the order of destruction.
    // assert(thread_pool);
    if(thread_pool) {
        thread_pool.reset();
    }
}

} // namespace fern
