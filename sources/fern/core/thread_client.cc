#include "fern/core/thread_client.h"
#include <cassert>
#include <iostream>
#include "fern/core/memory.h"


namespace fern {
namespace {

static std::unique_ptr<ThreadPool> thread_pool;

} // Anonymous namespace.


ThreadPool& ThreadClient::pool()
{
    // You did instantiate a ThreadClient instance, didn't you?
    assert(thread_pool);

    return *thread_pool;
}


ThreadClient::ThreadClient(
    size_t nr_threads)
{
    std::cout << "thread_client construct" << std::endl;
    // Instantiate the thread pool only once.
    // Note, that the pointer to the pool is created, but the pool itself
    // isn't. The pointer is a static variable.
    assert(!thread_pool);
    nr_threads = nr_threads > 0 ? nr_threads : 1;
    thread_pool = std::make_unique<ThreadPool>(nr_threads);
}


ThreadClient::~ThreadClient()
{
    std::cout << "thread_client destruct" << std::endl;
    // The pointer to the pool is a static variable.  The pool itself may or
    // may not be destructed already. This depends on whether we are static
    // or not and if so, the order of destruction.
    // assert(thread_pool);
    if(thread_pool) {
        thread_pool.reset();
    }
}

} // namespace fern
