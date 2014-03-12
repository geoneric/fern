#include "fern/core/thread_pool.h"


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


ThreadPool::ThreadPool()

    : _done(false),
      _joiner(_threads)

{
    size_t thread_count = std::thread::hardware_concurrency();

    try {
        for(size_t i = 0; i < thread_count; ++i) {
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


size_t ThreadPool::nr_threads() const
{
    return _threads.size();
}


void ThreadPool::worker_thread()
{
    while(!_done) {
        FunctionWrapper task;

        if(_work_queue.try_pop(task)) {
            task();
        }
        else {
            std::this_thread::yield();
        }
    }
}

} // namespace fern
