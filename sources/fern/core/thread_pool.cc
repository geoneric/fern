#include "fern/core/thread_pool.h"
#include <cassert>
#include <iostream>


namespace fern {

//! Construct a thread pool.
/*!
    \param         nr_threads Number of threads to create.
    \warning       As soon as the pool is created, the threads will start
                   probing the work_queue for work. Even if there are no tasks
                   for the threads to execute, they will add to the load of
                   the machine. In other words: only create a thread pool if
                   you have tasks that need to be done.
*/
ThreadPool::ThreadPool(
    size_t nr_threads)

    : _done(false),
      _joiner(_threads)

{
    std::cout << "thread_pool construct" << std::endl;
    assert(nr_threads > 0);

    try {
        for(size_t i = 0; i < nr_threads; ++i) {
            _threads.emplace_back(&ThreadPool::worker_thread, this);
        }
    }
    catch(...) {
        _done = true;
        throw;
    }
}


//! Destruct the thread pool.
/*!
*/
ThreadPool::~ThreadPool()
{
    std::cout << "thread_pool destruct" << std::endl;
    _done = true;
}


//! Return the number of threads in the pool.
/*!
*/
size_t ThreadPool::size() const
{
    return _threads.size();
}


//! Function that is run by each thread in the pool.
/*!
*/
void ThreadPool::worker_thread()
{
    while(!_done) {

        detail::FunctionWrapper task;

        if(_work_queue.try_pop(task)) {
            // Allright! We actually have something useful to do.
            task();
        }
        else {
            // No tasks in the queue, so hint the implementation to allow
            // other threads to run.
            std::this_thread::yield();
        }
    }
}

} // namespace fern
