#include "fern/core/thread_pool.h"
#include <cassert>
#include <iostream>


namespace fern {

FunctionWrapper::FunctionWrapper(
    FunctionWrapper&& other)

    : _concept(std::move(other._concept))

{
}


FunctionWrapper& FunctionWrapper::operator=(
    FunctionWrapper&& other)
{
    if(&other != this) {
        _concept = std::move(other._concept);
    }

    return *this;
}


void FunctionWrapper::operator()()
{
    _concept->call();
}


ThreadPool::ThreadPool(
    size_t nr_threads)

    : _done(false),
      _joiner(_threads)

{
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


ThreadPool::~ThreadPool()
{
    _done = true;
}


size_t ThreadPool::size() const
{
    return _threads.size();
}


void ThreadPool::worker_thread()
{
    while(!_done) {
        FunctionWrapper task;

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
