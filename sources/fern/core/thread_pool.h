#pragma once
#include <future>
#include <thread>
#include <vector>
#include "fern/core/detail/function_wrapper.h"
#include "fern/core/join_threads.h"
#include "fern/core/thread_safe_queue.h"


namespace fern {

//! Thread pool.
/*!
    A thread pool is like a collection of conveyor belts: you can put tasks on
    them or not, but the belts will always run and consume energy. Don't create
    a thread pool if you don't have anything useful for the threads to do.

    \sa            ThreadClient
*/
class ThreadPool
{

public:

                   ThreadPool          (size_t nr_threads);

                   ~ThreadPool         ();

    template<
        class Function>
    std::future<typename std::result_of<Function()>::type>
                   submit              (Function function);

    size_t         size                () const;

private:

    std::atomic_bool _done;

    ThreadSafeQueue<detail::FunctionWrapper> _work_queue;

    std::vector<std::thread> _threads;

    JoinThreads    _joiner;

    void           worker_thread       ();

};


//! Wrap the function or callable object in a task, push it on the work queue and return its future.
/*!
  \tparam    Function Type of function or callable object.
  \param     function Function or callable object to pass to thread pool.
  \return    Future holding the return value of the task, allowing the caller
             to wait for the task to complete.
*/
template<
    class Function>
inline std::future<typename std::result_of<Function()>::type>
    ThreadPool::submit(
        Function function)
{
    using result_type = typename std::result_of<Function()>::type;

    std::packaged_task<result_type()> task(std::move(function));
    std::future<result_type> result(task.get_future());
    _work_queue.push(std::move(task));

    return result;
}

} // namespace fern
