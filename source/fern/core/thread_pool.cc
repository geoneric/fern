// -----------------------------------------------------------------------------
// Fern © Geoneric
//
// This file is part of Geoneric Fern which is available under the terms of
// the GNU General Public License (GPL), version 2. If you do not want to
// be bound by the terms of the GPL, you may purchase a proprietary license
// from Geoneric (http://www.geoneric.eu/contact).
// -----------------------------------------------------------------------------
#include "fern/core/thread_pool.h"
#include <cassert>
#include <chrono>


namespace fern {

//! Construct a thread pool.
/*!
    \param         nr_threads Number of threads to create.
    \warning       As soon as the pool is created, the threads will start
                   probing the work queue for work.
*/
ThreadPool::ThreadPool(
    std::size_t nr_threads)

    : _done{false},
      _work_condition{},
      _mutex{},
      _work_queue{},
      _threads{},
      _joiner{_threads}

{
    assert(nr_threads > 0);

    try {
        for(std::size_t i = 0; i < nr_threads; ++i) {
            _threads.emplace_back(&ThreadPool::execute_task_or_wait, this);
        }
    }
    catch(...) {
        _done = true;
        _work_condition.notify_all();
        throw;
    }
}


//! Destruct the thread pool.
/*!
*/
ThreadPool::~ThreadPool()
{
    _done = true;
    _work_condition.notify_all();
}


//! Return the number of threads in the pool.
/*!
*/
std::size_t ThreadPool::size() const
{
    return _threads.size();
}


/// //! Function that is run by each thread in the pool.
/// /*!
///     If a task is available, it is executed. If not, the thread will yield.
/// 
///     This is useful if the work queue is hardly ever empty. All threads in the
///     pool will burn CPU cycles.
/// */
/// void ThreadPool::execute_task_or_yield()
/// {
///     while(!_done) {
/// 
///         detail::FunctionWrapper task;
/// 
///         if(_work_queue.try_pop(task)) {
///             // Allright! We actually have something useful to do.
///             task();
///         }
///         else {
///             // No tasks in the queue, so hint the implementation to allow
///             // other threads to run.
///             std::this_thread::yield();
///         }
///     }
/// }


//! Function that is run by each thread in the pool.
/*!
    If a task is available, it is executed. If not, the thread will wait for
    one to become available.

    This is useful if the work queue is empty regularly. Only the threads
    that are executing tasks are burning CPU cycles.
*/
void ThreadPool::execute_task_or_wait()
{
    using namespace std::chrono_literals;

    while(!_done) {

        // Check the queue for a task to execute, else wait for one to be
        // added.
        detail::FunctionWrapper task;

        if(_work_queue.try_pop(task)) {

            // Execute the task...
            task();
        }
        else {

            // Wait for a task to become available... The wait
            // automatically releases the mutex.

            // It seems that we can enter the wait while in fact the pool is
            // being destructed. The joiner will end up waiting for this
            // sleeping thread forever... By waking up once in a while, we
            // prevent this situation. Ideally, this should never happen...

            std::unique_lock<std::mutex> lock(_mutex);
            if(!_done) {
                _work_condition.wait_for(lock, 100ms);
            }
            // _work_condition.wait(lock);
        }
    }
}

} // namespace fern
